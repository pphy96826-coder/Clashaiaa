#define _GNU_SOURCE 1
#include "probe_features.h"
#include <android/log.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <link.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr const char *kLogTag = "NullsProbe";
constexpr uint16_t kPort = 26888;
constexpr uintptr_t kGameStateStepOffset = 0x010620a4;
constexpr uint8_t kGameStateStepPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};
constexpr uintptr_t kGameStateTickOffset = 0x01062bc0;
constexpr uintptr_t kGameAppSlotOffset = 0x1997cf0;

constexpr uintptr_t kControllerFullUpdateOffset = 0x00c88bbc;
constexpr uint8_t kControllerFullUpdatePrologue[16] = {
    0xff, 0x03, 0x02, 0xd1, 0xeb, 0x2b, 0x03, 0x6d, 0xe9, 0x23, 0x04, 0x6d, 0xfd, 0x7b, 0x05, 0xa9,
};

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, kLogTag, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kLogTag, __VA_ARGS__)

using GameStateStep = void (*)(void *);
using GameStateTick = int32_t (*)(void *);
using ControllerFullUpdateFn = void (*)(void *, float);

uintptr_t g_libg_base = 0;
GameStateStep g_original_game_state_step = nullptr;
GameStateTick g_game_state_tick = nullptr;
ControllerFullUpdateFn g_original_controller_full_update = nullptr;

std::atomic<void *> g_live_manager{nullptr};
std::atomic<int32_t> g_live_tick{-1};
std::atomic<bool> g_in_battle{false};
std::atomic<bool> g_controller_active{false};
// Start behind a lifecycle barrier.  A freshly loaded proxy can see the
// previous controller's state-8 callback before a new battle world exists;
// that callback must never make the lobby look active.
std::atomic<bool> g_live_expired{true};
// The host must explicitly arm a new lifecycle after a verified Battle tap
// or an operator-requested mid-battle attach.  State 8 and a manager pointer
// alone are not enough: the lobby/result controller can reuse them.
std::atomic<bool> g_context_armed{false};
std::atomic<void *> g_expired_manager{nullptr};
std::atomic<int32_t> g_expired_tick{-1};
std::atomic<int64_t> g_last_step_time_ms{0};
std::atomic<bool> g_init_worker_started{false};

// Buffer holding latest JSON state
pthread_mutex_t g_state_mutex = PTHREAD_MUTEX_INITIALIZER;
char g_latest_state_json[524288] = "{\"in_battle\":false}\n";

template <typename T>
inline T read_field(const void *obj, size_t offset) {
    if (obj == nullptr) return T{};
    T val{};
    // MuMu's ARM64 runtime may attach a top-byte allocation tag to native
    // pointers (for example 0xb4...).  TBI makes the tagged pointer usable by
    // the CPU, but arithmetic/validation must compare the untagged address.
    const uintptr_t untagged = reinterpret_cast<uintptr_t>(obj) & 0x00ffffffffffffffULL;
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(untagged) + offset;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}

inline bool is_valid_pointer(const void *ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) & 0x00ffffffffffffffULL;
    return ptr != nullptr && addr >= 0x10000ULL && addr <= 0x0000ffffffffffffULL;
}

// The proxy is loaded while Android is still resolving the game's native
// libraries.  A fixed libg offset is only usable after its mapping is present;
// probing it with a raw dereference during that window can take down the whole
// client.  Keep startup reads inside a mapped, readable range.
bool is_readable_range(const void *address, size_t size) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
    if (begin == 0 || size == 0 || begin + size < begin) return false;

    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return false;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long long map_begin = 0;
        unsigned long long map_end = 0;
        char permissions[5] = {};
        if (sscanf(line, "%llx-%llx %4s", &map_begin, &map_end, permissions) == 3 &&
            permissions[0] == 'r' && begin >= map_begin && begin + size <= map_end) {
            fclose(maps);
            return true;
        }
    }
    fclose(maps);
    return false;
}

bool read_pointer_if_mapped(const void *address, void **value) {
    if (value == nullptr || !is_readable_range(address, sizeof(void *))) return false;
    std::memcpy(value, address, sizeof(void *));
    return is_valid_pointer(*value);
}

int64_t current_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

struct LibgLookup {
    uintptr_t base = 0;
};

int find_libg_callback(dl_phdr_info *info, size_t, void *opaque) {
    if (info == nullptr || info->dlpi_name == nullptr || strstr(info->dlpi_name, "libg.so") == nullptr) {
        return 0;
    }
    auto *lookup = static_cast<LibgLookup *>(opaque);
    lookup->base = static_cast<uintptr_t>(info->dlpi_addr);
    return 1;
}

uintptr_t find_libg_base_from_proc_maps() {
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return 0;
    char line[512];
    uintptr_t base = 0;
    while (fgets(line, sizeof(line), maps)) {
        if (strstr(line, "libg.so") && strstr(line, "r--p") && strstr(line, "00000000")) {
            sscanf(line, "%lx-", &base);
            break;
        }
    }
    fclose(maps);
    return base;
}

uintptr_t find_libg_base() {
    // Prefer the loader's own module table once JNI_OnLoad has returned.  The
    // proc-maps fallback keeps this compatible with images whose translated
    // loader does not publish libg through dl_iterate_phdr.
    LibgLookup lookup;
    dl_iterate_phdr(find_libg_callback, &lookup);
    return lookup.base != 0 ? lookup.base : find_libg_base_from_proc_maps();
}

void emit_absolute_jump(void *destination, const void *target) {
    // ldr x17, #8; br x17; .quad target
    const uint32_t instructions[2] = {0x58000051u, 0xd61f0220u};
    std::memcpy(destination, instructions, sizeof(instructions));
    std::memcpy(static_cast<uint8_t *>(destination) + 8, &target, sizeof(target));
}

bool make_page_writable(void *address, int protection) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) page_size = 4096;
    uintptr_t raw = reinterpret_cast<uintptr_t>(address);
    uintptr_t page = raw & ~static_cast<uintptr_t>(page_size - 1);
    return mprotect(reinterpret_cast<void *>(page), static_cast<size_t>(page_size), protection) == 0;
}

bool install_inline_hook(uintptr_t target_addr, const uint8_t *expected_prologue, size_t prologue_len,
                         const void *hook_fn, void **out_original, const char *name) {
    uint8_t *target = reinterpret_cast<uint8_t *>(target_addr);
    if (!is_readable_range(target, prologue_len)) {
        LOGE("Target for %s is not readable at %p", name, target);
        return false;
    }
    if (std::memcmp(target, expected_prologue, prologue_len) != 0) {
        LOGE("Prologue mismatch for %s at %p", name, target);
        return false;
    }

    void *trampoline = mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (trampoline == MAP_FAILED) {
        LOGE("mmap trampoline failed for %s", name);
        return false;
    }

    std::memcpy(trampoline, target, prologue_len);
    emit_absolute_jump(static_cast<uint8_t *>(trampoline) + prologue_len, target + prologue_len);
    __builtin___clear_cache(static_cast<char *>(trampoline), static_cast<char *>(trampoline) + 32);

    if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
        LOGE("mprotect target failed for %s", name);
        return false;
    }

    *out_original = trampoline;
    emit_absolute_jump(target, hook_fn);
    __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
    make_page_writable(target, PROT_READ | PROT_EXEC);

    LOGI("%s hook successfully installed at %p -> trampoline %p", name, target, trampoline);
    return true;
}

#include "runtime_snapshot.inc"
#include "card_runtime.inc"
#include "ability_runtime.inc"
#include "causal_deployment.inc"
#include "spawn_relations.inc"
#include "impact_events.inc"
#include "damage_events.inc"
#include "effect_origin.inc"
#include "heal_events.inc"
#if CR_EXPERIMENTAL_PRODUCER_ORIGINS
#include "area_origin.inc"
#include "projectile_origin.inc"
#endif

bool effect_origin_runtime(uint64_t &epoch) {
    epoch=0;void *world=nullptr,*manager=nullptr;int tick=-1;
    if (!g_effect_origin_ready || !deployment_context(world,manager,tick)) return false;
    effect_origin_epoch(world,tick);epoch=g_effect_origin_epoch;
    return epoch>0;
}
bool append_live_effect_origin(char *json,int &offset,size_t size,void *entry,void *target) {
    if(offset<0 || size_t(offset)+1024>=size)return false;
    offset+=snprintf(json+offset,size-offset,",\"origin\":");
    void *world=nullptr,*manager=nullptr;int tick=-1;EffectOrigin origin;
    if (g_effect_origin_ready && deployment_context(world,manager,tick) &&
        effect_origin_lookup(entry,target,world,manager,tick,origin))
        append_effect_origin(json,offset,size,origin);
    else offset+=snprintf(json+offset,size-offset,"null");
    return true;
}

void publish_live_capture_error(int32_t tick, const char *reason) {
    char json[1024];
    snprintf(json, sizeof(json),
        "{\"schema\":\"nulls-live.v3\",\"in_battle\":true,\"tick\":%d,"
        "\"capture_complete\":false,\"capture_error\":\"%s\","
        "\"players\":[],\"entities\":[],\"entities_complete\":false}\n",
        tick, reason ? reason : "unknown");
    pthread_mutex_lock(&g_state_mutex);
    std::memcpy(g_latest_state_json, json, strlen(json) + 1);
    pthread_mutex_unlock(&g_state_mutex);
}

// Return whether the battle world/player context is real enough to classify
// the session as active.  An active world may still have an incomplete entity
// capture; that remains fail-closed in the payload, but must not be confused
// with a stale state-8 controller left behind by the result screen.
bool extract_live_state(void *manager, int32_t tick) {
    if (!manager) {
        publish_live_capture_error(tick, "manager_unavailable");
        return false;
    }
    void *world = read_field<void *>(manager, 0xa8);
    if (!is_valid_pointer(world)) {
        publish_live_capture_error(tick, "world_unavailable");
        return false;
    }
    deployment_epoch(world,tick);
    effect_origin_epoch(world,tick);
#if CR_EXPERIMENTAL_PRODUCER_ORIGINS
    area_origin_epoch(world,tick);
#endif
    pthread_mutex_lock(&g_attack_mutex);
    attack_epoch_locked(world,tick);
    pthread_mutex_unlock(&g_attack_mutex);

    // Upstream read_battle_result: game simulation ticks may continue during
    // result animation. Do not use tick freshness as proof that input is open.
    uint8_t finalized=255;
    int32_t world_result=-1;
    bool result_known=safe_field(world,0x1e0,finalized) && finalized<=1 &&
                      safe_field(world,0x1b8,world_result);

    void *player0 = read_field<void *>(world, 0xe0);
    void *player1 = read_field<void *>(world, 0xe8);
    if (!is_valid_pointer(player1) && !is_valid_pointer(player0)) {
        publish_live_capture_error(tick, "players_unavailable");
        return false;
    }

    char json[524288];
    int offset = snprintf(json, sizeof(json),
        "{\"schema\":\"nulls-live.v3\",\"in_battle\":true,\"tick\":%d,\"battle_result\":{\"validated\":%s,\"finalized\":%s,\"world_result_raw\":%d},\"hook_diagnostics\":{\"release_calls\":%llu,\"release_caller\":%llu,\"release_result\":%d,\"release_owner_id\":%u},\"players\":[", tick,
        result_known?"true":"false",result_known && finalized?"true":"false",world_result,
        static_cast<unsigned long long>(g_execute_calls.load()),static_cast<unsigned long long>(g_execute_caller.load()),g_execute_result.load(),g_execute_owner_id.load());

    // Helper to format each player (owner 0 and owner 1)
    for (int owner = 0; owner < 2; ++owner) {
        void *p = (owner == 0) ? player0 : player1;
        void *identity = read_field<void *>(world, 0x30 + owner * sizeof(void *));
        uint32_t id_high = 0, id_low = 0;
        if (is_valid_pointer(identity)) {
            id_high = read_field<uint32_t>(identity, 0x00);
            id_low = read_field<uint32_t>(identity, 0x04);
        }
        uint64_t account_id = (static_cast<uint64_t>(id_high) << 32) | id_low;

        int32_t elixir_raw = 0;
        float elixir = 0.0f;
        if (is_valid_pointer(p)) {
            elixir_raw = read_field<int32_t>(p, 0x2f8);
            elixir = static_cast<float>(elixir_raw) / 10000.0f;
            if (elixir < 0.0f) elixir = 0.0f;
            if (elixir > 10.0f) elixir = 10.0f;
        }

        // Deck
        void *deck = read_field<void *>(world, 0x88 + owner * sizeof(void *));
        void **slots = nullptr;
        int32_t deck_count = 0;
        if (is_valid_pointer(deck)) {
            slots = read_field<void **>(deck, 0x20);
            deck_count = read_field<int32_t>(deck, 0x2c);
        }

        offset += snprintf(json + offset, sizeof(json) - offset,
            "%s{\"owner\":%d,\"accountId\":%llu,\"elixir\":%.2f,\"elixir_raw\":%d,\"hand\":[",
            owner == 0 ? "" : ",", owner, static_cast<unsigned long long>(account_id), elixir, elixir_raw);

        // Hand
        if (is_valid_pointer(p) && is_valid_pointer(slots) && deck_count > 0) {
            const int32_t *hand_indices = read_field<const int32_t *>(p, 0x220);
            int32_t hand_count = read_field<int32_t>(p, 0x22c);
            if (is_valid_pointer(hand_indices) && hand_count > 0 && hand_count <= 8) {
                bool first = true;
                for (int i = 0; i < hand_count && i < 4; ++i) {
                    int32_t slot_idx = hand_indices[i];
                    int32_t card_id = 0;
                    if (slot_idx >= 0 && slot_idx < deck_count && is_valid_pointer(slots[slot_idx])) {
                        void *card_data = read_field<void *>(slots[slot_idx], 0x10);
                        if (is_valid_pointer(card_data)) {
                            card_id = read_field<int32_t>(card_data, 0x40);
                        }
                    }
                    offset += snprintf(json + offset, sizeof(json) - offset,
                        "%s{\"slot\":%d,\"card_id\":%d}", first ? "" : ",", i, card_id);
                    first = false;
                }
            }
        }

        // Exact ordered native cycle (same verified layout as FirstLight).
        offset += snprintf(json + offset, sizeof(json) - offset, "],\"cycle\":[");
        if (is_valid_pointer(p) && is_valid_pointer(slots) && deck_count > 0 && deck_count <= 16) {
            const int32_t *cycle = read_field<const int32_t *>(p, 0x230);
            int32_t cycle_count = read_field<int32_t>(p, 0x23c);
            if (is_valid_pointer(cycle) && cycle_count >= 0 && cycle_count <= 8) {
                for (int i = 0; i < cycle_count; ++i) {
                    int32_t idx = cycle[i];
                    int32_t cid = 0;
                    if (idx >= 0 && idx < deck_count && is_valid_pointer(slots[idx])) {
                        void *data = read_field<void *>(slots[idx], 0x10);
                        if (is_valid_pointer(data)) cid = read_field<int32_t>(data, 0x40);
                    }
                    offset += snprintf(json + offset, sizeof(json) - offset, "%s%d", i ? "," : "", cid);
                }
            }
        }

        // Full Deck
        offset += snprintf(json + offset, sizeof(json) - offset, "],\"deck\":[");
        if (is_valid_pointer(slots) && deck_count > 0 && deck_count <= 16) {
            for (int d = 0; d < deck_count && d < 8; ++d) {
                int32_t d_card_id = 0;
                if (is_valid_pointer(slots[d])) {
                    void *card_data = read_field<void *>(slots[d], 0x10);
                    if (is_valid_pointer(card_data)) {
                        d_card_id = read_field<int32_t>(card_data, 0x40);
                    }
                }
                offset += snprintf(json + offset, sizeof(json) - offset, "%s%d", d == 0 ? "" : ",", d_card_id);
            }
        }
        offset += snprintf(json + offset, sizeof(json) - offset, "]");
        append_card_runtime(json,offset,sizeof(json),p,slots,deck_count);
        append_ability_runtime(json,offset,sizeof(json),p,owner);
        offset += snprintf(json + offset, sizeof(json) - offset, "}");
    }

    offset += snprintf(json + offset, sizeof(json) - offset, "],\"entities\":[");

    // Extract arena entities from object_manager (player0 + 0x10)
    bool king_alive[2] = {false, false};
    int princess_alive_count[2] = {0, 0};
    bool entities_complete = false;

    if (is_valid_pointer(player0)) {
        void *object_manager = read_field<void *>(player0, 0x10);
        if (is_valid_pointer(object_manager)) {
            void **objects = read_field<void **>(object_manager, 0x08);
            int32_t count = read_field<int32_t>(object_manager, 0x14);
            if (is_valid_pointer(objects) && count > 0 && count < 600) {
                entities_complete = true;
                bool first_ent = true;
                for (int i = 0; i < count; ++i) {
                    void *obj = objects[i];
                    if (!is_valid_pointer(obj)) { entities_complete = false; continue; }

                    uint32_t native_id = read_field<uint32_t>(obj, 0x08);
                    int32_t owner = read_field<int32_t>(obj, 0x78);
                    int32_t px = read_field<int32_t>(obj, 0x7c);
                    int32_t py = read_field<int32_t>(obj, 0x80);
                    int32_t card_id = read_field<int32_t>(obj, 0xac);
                    void *object_data = read_field<void *>(obj, 0x48);
                    uint32_t data_global_id = is_valid_pointer(object_data) ? read_field<uint32_t>(object_data, 0x40) : 0;

                    // Component 2 is Hitpoints component
                    int32_t hp = 0, max_hp = 0;
                    void **components = read_field<void **>(obj, 0x18);
                    int32_t comp_count = read_field<int32_t>(obj, 0x24);
                    if (comp_count > 2 && is_valid_pointer(components) && is_valid_pointer(components[2])) {
                        void *comp2 = components[2];
                        if (read_field<void *>(comp2, 0x08) == obj) {
                            hp = read_field<int32_t>(comp2, 0x10);
                            max_hp = read_field<int32_t>(comp2, 0x14);
                        }
                    }

                    // Check towers
                    if (card_id == -1 && owner >= 0 && owner < 2) {
                        if (px == 9000 && (py == 3000 || py == 29000)) {
                            king_alive[owner] = (hp > 0);
                        } else if ((px == 3500 || px == 14500) && (py == 6500 || py == 25500) && hp > 0) {
                            ++princess_alive_count[owner];
                        }
                    }

                    if (offset + 8192 < (int)sizeof(json)) {
                        offset += snprintf(json + offset, sizeof(json) - offset,
                            "%s{\"id\":%u,\"owner\":%d,\"x\":%d,\"y\":%d,\"card_id\":%d,\"hp\":%d,\"max_hp\":%d,\"native_data_global_id\":%u",
                            first_ent ? "" : ",", native_id, owner, px, py, card_id, hp, max_hp, data_global_id);
                        bool is_tower = card_id == -1 && (px == 9000 || px == 3500 || px == 14500) &&
                            (py == 3000 || py == 29000 || py == 6500 || py == 25500);
                        append_runtime_snapshot(json, offset, sizeof(json), obj, objects, count, is_tower);
                        append_deployment_member(json,offset,sizeof(json),obj);
                        append_spawn_relation(json,offset,sizeof(json),obj);
#if CR_EXPERIMENTAL_PRODUCER_ORIGINS
                        append_area_origin(json,offset,sizeof(json),obj);
                        append_projectile_origin(json,offset,sizeof(json),obj);
#endif
                        offset += snprintf(json+offset,sizeof(json)-offset,"}");
                        first_ent = false;
                    } else { entities_complete = false; break; }
                }
            }
        }
    }

    // Top-level convenience fields for local player (owner 0)
    void *local_p = is_valid_pointer(player0) ? player0 : player1;
    int32_t local_elixir_raw = local_p ? read_field<int32_t>(local_p, 0x2f8) : 0;
    float local_elixir = static_cast<float>(local_elixir_raw) / 10000.0f;
    if (local_elixir < 0.0f) local_elixir = 0.0f;
    if (local_elixir > 10.0f) local_elixir = 10.0f;

    offset += snprintf(json + offset, sizeof(json) - offset, "]");
    append_client_input_edges(json,offset,sizeof(json),world,tick);
    append_deployment_diagnostics(json,offset,sizeof(json));
    append_spawn_diagnostics(json,offset,sizeof(json));
#if CR_EXPERIMENTAL_PRODUCER_ORIGINS
    append_area_origin_diagnostics(json,offset,sizeof(json));
    append_projectile_origin_diagnostics(json,offset,sizeof(json),world,tick);
#endif
    append_impact_events(json,offset,sizeof(json),world,tick);
    append_damage_events(json,offset,sizeof(json),world,tick);
    append_heal_events(json,offset,sizeof(json),world,tick);
    snprintf(json + offset, sizeof(json) - offset,
        ",\"entities_complete\":%s,\"capture_complete\":%s,\"local_owner\":null,\"elixir\":%.2f,\"elixir_raw\":%d,\"crowns\":[%d,%d]}\n",
        entities_complete ? "true" : "false", entities_complete ? "true" : "false", local_elixir, local_elixir_raw,
        king_alive[1] ? 2 - princess_alive_count[1] : 3, king_alive[0] ? 2 - princess_alive_count[0] : 3);

    pthread_mutex_lock(&g_state_mutex);
    std::memcpy(g_latest_state_json, json, strlen(json) + 1);
    pthread_mutex_unlock(&g_state_mutex);
    return true;
}

void hooked_game_state_step(void *manager) {
    // GameStateManager::step is also used during result/lobby teardown and
    // replay-style processing.  The live endpoint is therefore maintained by
    // BattleController::fullUpdate only; this hook must never refresh its
    // activity timestamp or resurrect a completed battle.

    // Call original step
    g_original_game_state_step(manager);
}

void hooked_controller_full_update(void *controller, float dt) {
    if (controller != nullptr) {
        int32_t state = read_field<int32_t>(controller, 0x30);
        static int32_t s_last_state = -1;
        if (state != s_last_state) {
            LOGI("BattleController state change: %d -> %d (controller=%p)", s_last_state, state, controller);
            s_last_state = state;
        }

        // State 8 is active in-battle state.  Clear the exported live marker
        // as soon as the controller leaves it; otherwise the TCP endpoint can
        // keep serving the last battle frame after result/lobby teardown.
        if (state == 8) {
            void *manager = read_field<void *>(controller, 0x90);
            static bool s_logged_first_active_manager = false;
            if (!s_logged_first_active_manager) {
                LOGI("active controller=%p manager=%p valid=%d", controller, manager,
                     is_valid_pointer(manager) ? 1 : 0);
                s_logged_first_active_manager = true;
            }
            // Some result-screen builds do not publish the controller's
            // non-active state before the next battle.  In that case the
            // transport timeout correctly expires the old manager, but the
            // old expired flag would otherwise poison every later battle.
            // Re-arm only when the live tick strictly falls.  A result/lobby
            // transition can replace the manager while retaining the last
            // battle tick, so manager identity alone is not evidence of a
            // new battle.  A lower tick covers both a reused manager and a
            // newly allocated manager without resurrecting the old frame.
            int32_t tick = g_game_state_tick ? g_game_state_tick(manager) : -1;
            const void *expired_manager = g_expired_manager.load(std::memory_order_acquire);
            const int32_t expired_tick = g_expired_tick.load(std::memory_order_acquire);
            void *candidate_world = nullptr;
            if (is_valid_pointer(manager)) {
                candidate_world = read_field<void *>(manager, 0xa8);
            }
            const bool world_ready = is_valid_pointer(candidate_world);
            const bool manager_changed = expired_manager == nullptr || manager != expired_manager;
            const bool new_context = g_context_armed.load(std::memory_order_acquire) && tick >= 0 &&
                ((expired_tick >= 0 && tick < expired_tick) ||
                 (world_ready && manager_changed));
            if (is_valid_pointer(manager) && g_live_expired.load(std::memory_order_acquire) && new_context) {
                g_live_expired.store(false, std::memory_order_release);
                g_expired_manager.store(nullptr, std::memory_order_release);
                g_expired_tick.store(-1, std::memory_order_release);
                g_live_tick.store(-1, std::memory_order_release);
            }
            if (is_valid_pointer(manager) && g_context_armed.load(std::memory_order_acquire) &&
                !g_live_expired.load(std::memory_order_acquire)) {
                g_controller_active.store(true, std::memory_order_release);
                if (tick != g_live_tick.load(std::memory_order_acquire)) {
                    // Keep the controller/session active while the world is
                    // being rebuilt.  extract_live_state() publishes an
                    // incomplete, fail-closed frame in that window; treating
                    // it as lobby here breaks mid-battle attach because a
                    // real active battle can temporarily expose no world.
                    // The expiry barrier above and the transport timeout
                    // still prevent a stale result-screen manager from being
                    // reused for writes.
                    extract_live_state(manager, tick);
                    g_live_manager.store(manager, std::memory_order_release);
                    g_live_tick.store(tick, std::memory_order_release);
                    g_last_step_time_ms.store(current_time_ms(), std::memory_order_release);
                    g_in_battle.store(true, std::memory_order_release);
                }
            } else {
                g_controller_active.store(false, std::memory_order_release);
                g_in_battle.store(false, std::memory_order_release);
                g_live_manager.store(nullptr, std::memory_order_release);
                g_live_tick.store(-1, std::memory_order_release);
                g_last_step_time_ms.store(0, std::memory_order_release);
            }
        } else {
            g_controller_active.store(false, std::memory_order_release);
            // The controller may leave state 8 before the host asks for one
            // more frame.  Preserve the same fail-closed boundary as the
            // transport-timeout path so the old manager cannot be published
            // again from the result/lobby transition.
            void *previous_manager = g_live_manager.load(std::memory_order_acquire);
            int32_t previous_tick = g_live_tick.load(std::memory_order_acquire);
            if (g_in_battle.load(std::memory_order_acquire) || previous_manager != nullptr) {
                g_expired_manager.store(previous_manager, std::memory_order_release);
                g_expired_tick.store(previous_tick, std::memory_order_release);
                g_live_expired.store(true, std::memory_order_release);
            }
            g_in_battle.store(false, std::memory_order_release);
            g_live_manager.store(nullptr, std::memory_order_release);
            g_live_tick.store(-1, std::memory_order_release);
            g_last_step_time_ms.store(0, std::memory_order_release);
        }
    }

    if (g_original_controller_full_update) {
        g_original_controller_full_update(controller, dt);
    }
}

void *tcp_server_thread(void *) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        LOGE("Failed to create socket: errno=%d", errno);
        return nullptr;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(kPort);

    if (bind(server_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        LOGE("Failed to bind port %u: errno=%d", kPort, errno);
        close(server_fd);
        return nullptr;
    }

    if (listen(server_fd, 5) < 0) {
        LOGE("Failed to listen: errno=%d", errno);
        close(server_fd);
        return nullptr;
    }

    LOGI("TCP server listening on port %u", kPort);

    while (true) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, reinterpret_cast<struct sockaddr *>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        char req_buf[256];
        ssize_t n = recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
        if (n > 0) {
            req_buf[n] = '\0';
            if (strncmp(req_buf, "RESET", 5) == 0) {
                // Explicit lifecycle boundary from the host.  This only
                // invalidates probe-owned live state; it never touches the
                // game or submits an input/action.
                // Keep an old manager/tick as an expiry barrier.  A result
                // screen can leave the controller reporting state 8 for a
                // while; clearing the barrier here would resurrect that
                // stale manager on the next full-update callback.  The
                // barrier is released only by the lower tick of a new battle.
                void *prior_manager = g_live_manager.load(std::memory_order_acquire);
                const int32_t prior_tick = g_live_tick.load(std::memory_order_acquire);
                const bool prior_expired = g_live_expired.load(std::memory_order_acquire);
                // RESET always begins behind a barrier, including when no
                // prior live manager was published.  This is the important
                // cold-start case: the old state-8 controller may still be
                // calling us while the game is visibly in the lobby.
                if (prior_manager != nullptr || prior_tick >= 0 || prior_expired) {
                    g_expired_manager.store(prior_manager, std::memory_order_release);
                    g_expired_tick.store(prior_tick, std::memory_order_release);
                } else {
                    g_expired_manager.store(nullptr, std::memory_order_release);
                    g_expired_tick.store(-1, std::memory_order_release);
                }
                g_live_expired.store(true, std::memory_order_release);
                g_context_armed.store(false, std::memory_order_release);
                g_controller_active.store(false, std::memory_order_release);
                g_in_battle.store(false, std::memory_order_release);
                g_live_manager.store(nullptr, std::memory_order_release);
                g_live_tick.store(-1, std::memory_order_release);
                g_last_step_time_ms.store(0, std::memory_order_release);
                pthread_mutex_lock(&g_state_mutex);
                std::strcpy(g_latest_state_json, "{\"in_battle\":false}\n");
                pthread_mutex_unlock(&g_state_mutex);
                const char *reset_json = "{\"ok\":true,\"reset\":true}\n";
                send(client_fd, reset_json, std::strlen(reset_json), 0);
            } else if (strncmp(req_buf, "ARM", 3) == 0) {
                // This only changes the probe lifecycle gate.  It never
                // taps the game and never queues a native command.
                g_context_armed.store(true, std::memory_order_release);
                // Keep the expiry barrier established by RESET/result
                // transition.  The callback must observe a lower tick or a
                // different ready manager before a new battle is published.
                // Clearing this barrier here would resurrect a stale result
                // screen manager immediately after the Battle tap.
                g_live_manager.store(nullptr, std::memory_order_release);
                g_live_tick.store(-1, std::memory_order_release);
                g_last_step_time_ms.store(0, std::memory_order_release);
                const char *arm_json = "{\"ok\":true,\"armed\":true}\n";
                send(client_fd, arm_json, std::strlen(arm_json), 0);
            } else if (strncmp(req_buf, "ATTACH", 6) == 0) {
                // Explicit operator-authorized mid-battle attach.  Unlike
                // ARM-after-Battle, this intentionally adopts the currently
                // published ready manager even when its tick is already past
                // the old expiry tick.  This is still control-plane only:
                // no game input or native command is queued here.
                g_context_armed.store(true, std::memory_order_release);
                g_live_expired.store(false, std::memory_order_release);
                g_expired_manager.store(nullptr, std::memory_order_release);
                g_expired_tick.store(-1, std::memory_order_release);
                g_live_manager.store(nullptr, std::memory_order_release);
                g_live_tick.store(-1, std::memory_order_release);
                g_last_step_time_ms.store(0, std::memory_order_release);
                const char *attach_json = "{\"ok\":true,\"attached\":true}\n";
                send(client_fd, attach_json, std::strlen(attach_json), 0);
            } else if (strncmp(req_buf, "PING", 4) == 0) {
                send(client_fd, "PONG\n", 5, 0);
            } else {
                int64_t now = current_time_ms();
                bool battle_active = g_in_battle.load(std::memory_order_acquire) &&
                                     (now - g_last_step_time_ms.load(std::memory_order_acquire) < 1000);
                // A timeout is not a lifecycle transition. Hide the expired
                // frame, but retain the last tick so only a NEW controller
                // capture can make it fresh again. RESET and controller
                // teardown still enforce their explicit expiry barriers.

                pthread_mutex_lock(&g_state_mutex);
                if (battle_active) {
                    send(client_fd, g_latest_state_json, strlen(g_latest_state_json), 0);
                } else {
                    const char *idle_json = g_in_battle.load(std::memory_order_acquire)
                        ? "{\"in_battle\":false,\"status\":\"stale\"}\n"
                        : "{\"in_battle\":false}\n";
                    send(client_fd, idle_json, strlen(idle_json), 0);
                }
                pthread_mutex_unlock(&g_state_mutex);
            }
        }
        close(client_fd);
    }
    close(server_fd);
    return nullptr;
}

void *init_worker_thread(void *) {
    LOGI("NullsProbe init worker started, waiting for GameApp initialization...");
    // Do not inspect libg while the real SDK's JNI_OnLoad is still resolving
    // the native bridge.  This mirrors the delayed bootstrap used by the
    // existing FirstLight probe on translated Android runtimes.
    usleep(2000000);
    uintptr_t libg_base = 0;
    // GameApp can be recreated during login/lobby startup and may take longer
    // than the old 30-second one-shot window on a cold MuMu boot.  Do not
    // permanently kill the probe in that window: keep the worker dormant and
    // retry until a valid instance is published.  No hooks or listener are
    // exposed before this gate passes, so the host cannot mistake a partial
    // initialization for a healthy runtime.
    unsigned int retry_log = 0;
    for (;;) {
        libg_base = 0;
        for (int i = 0; i < 300 && libg_base == 0; ++i) { // 30s per scan
            libg_base = find_libg_base();
            if (libg_base != 0) {
                void *app_ptr = nullptr;
                if (read_pointer_if_mapped(reinterpret_cast<void *>(libg_base + kGameAppSlotOffset), &app_ptr)) {
                    LOGI("GameApp initialized at %p (libg=%p)", app_ptr, reinterpret_cast<void *>(libg_base));
                    break;
                }
                libg_base = 0;
            }
            usleep(100000); // 100ms
        }
        if (libg_base != 0) break;
        if ((retry_log++ % 2) == 0) {
            LOGI("GameApp initialization still pending; retrying native bootstrap");
        }
        usleep(1000000);
    }

    // Give an extra 500ms safety buffer
    usleep(500000);

    g_libg_base = libg_base;
    g_game_state_tick = reinterpret_cast<GameStateTick>(libg_base + kGameStateTickOffset);
    g_attack_hooks_ready=install_attack_edges();
    g_client_input_hook_ready=install_client_input_observer();
    g_deployment_hooks_ready=install_deployment_observers();
    g_summon_hooks_ready=install_spawn_observers();
    g_impact_ready=install_impact_observer();
    g_damage_ready=install_damage_observer();
    g_effect_origin_ready=g_attack_hooks_ready && install_effect_origin_observers();
    g_heal_ready=install_heal_observer();
#if CR_EXPERIMENTAL_PRODUCER_ORIGINS
    g_area_origin_ready=g_deployment_hooks_ready && install_area_origin_observers();
    if (g_area_origin_ready) g_area_relation_observer=area_origin_registered;
    g_projectile_origin_ready=g_summon_hooks_ready && g_deployment_hooks_ready && install_projectile_origin_observers();
    if (g_projectile_origin_ready) {
        g_projectile_queued_observer=projectile_origin_queued;
        g_projectile_relation_observer=projectile_origin_registered;
    }
#endif
    LOGI("Attack edge hooks ready: %d",g_attack_hooks_ready);

    // 1. Hook GameStateManager::step (offline replays & card preview micro-sims)
    install_inline_hook(libg_base + kGameStateStepOffset, kGameStateStepPrologue, 16,
                        reinterpret_cast<const void *>(&hooked_game_state_step),
                        reinterpret_cast<void **>(&g_original_game_state_step),
                        "GameStateManager::step");

    // 2. Hook BattleController::fullUpdate (online PvP live battle engine at 60 Hz)
    install_inline_hook(libg_base + kControllerFullUpdateOffset, kControllerFullUpdatePrologue, 16,
                        reinterpret_cast<const void *>(&hooked_controller_full_update),
                        reinterpret_cast<void **>(&g_original_controller_full_update),
                        "BattleController::fullUpdate");

    LOGI("NullsProbe hooks installed successfully!");

    pthread_t tcp_thread;
    pthread_create(&tcp_thread, nullptr, tcp_server_thread, nullptr);
    pthread_detach(tcp_thread);
    return nullptr;
}

} // namespace

#define DECL_FORWARD(name) \
    __attribute__((visibility("hidden"))) void *g_sym_##name = nullptr; \
    extern "C" __attribute__((naked, visibility("default"))) void name() { \
        __asm__ volatile ( \
            "adrp x16, g_sym_" #name "\n" \
            "ldr x16, [x16, :lo12:g_sym_" #name "]\n" \
            "br x16\n" \
        ); \
    }

DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_avatarImageData)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_bindAccount)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_forgetAccount)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_generateQrCodeResult)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_getConfig)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_getSessionTokenResult)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_loadAccount)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_logOut)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_publicProfileData)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_setProfile)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_setProfileFailed)
DECL_FORWARD(Java_xyz_daniillnull_connect_sdk_NTConnect_windowDidDismiss)
DECL_FORWARD(_ZN11SupercellIdC1ERK17SupercellIdParams)
DECL_FORWARD(_ZN11SupercellIdC2ERK17SupercellIdParams)
DECL_FORWARD(_ZN11SupercellIdD0Ev)
DECL_FORWARD(_ZN11SupercellIdD1Ev)
DECL_FORWARD(_ZN11SupercellIdD2Ev)
DECL_FORWARD(_ZN15SupercellSocialC1EP23SupercellSocialDelegate)
DECL_FORWARD(_ZN15SupercellSocialC2EP23SupercellSocialDelegate)
DECL_FORWARD(_ZN15SupercellSocialD0Ev)
DECL_FORWARD(_ZN15SupercellSocialD1Ev)
DECL_FORWARD(_ZN15SupercellSocialD2Ev)

static void *g_real_scid_handle = nullptr;

static bool resolve_real_symbols() {
    if (g_real_scid_handle) return true;

    g_real_scid_handle = dlopen("libscid_sdk_real.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_real_scid_handle) {
        FILE *maps = fopen("/proc/self/maps", "r");
        if (maps) {
            char line[512];
            while (fgets(line, sizeof(line), maps)) {
                if (strstr(line, "libscid_sdk.so")) {
                    char *path = strchr(line, '/');
                    if (path) {
                        char real_path[512];
                        strncpy(real_path, path, sizeof(real_path) - 1);
                        real_path[sizeof(real_path) - 1] = '\0';
                        char *nl = strchr(real_path, '\n');
                        if (nl) *nl = '\0';
                        char *end = strstr(real_path, "libscid_sdk.so");
                        if (end) {
                            strcpy(end, "libscid_sdk_real.so");
                            g_real_scid_handle = dlopen(real_path, RTLD_NOW | RTLD_GLOBAL);
                            if (g_real_scid_handle) break;
                        }
                    }
                }
            }
            fclose(maps);
        }
    }

    if (!g_real_scid_handle) {
        LOGE("FATAL: dlopen libscid_sdk_real.so failed: %s", dlerror());
        return false;
    }

    LOGI("libscid_sdk_real.so opened successfully at %p", g_real_scid_handle);

#define RESOLVE(name) \
    g_sym_##name = dlsym(g_real_scid_handle, #name);

    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_avatarImageData)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_bindAccount)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_forgetAccount)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_generateQrCodeResult)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_getConfig)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_getSessionTokenResult)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_loadAccount)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_logOut)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_publicProfileData)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_setProfile)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_setProfileFailed)
    RESOLVE(Java_xyz_daniillnull_connect_sdk_NTConnect_windowDidDismiss)
    RESOLVE(_ZN11SupercellIdC1ERK17SupercellIdParams)
    RESOLVE(_ZN11SupercellIdC2ERK17SupercellIdParams)
    RESOLVE(_ZN11SupercellIdD0Ev)
    RESOLVE(_ZN11SupercellIdD1Ev)
    RESOLVE(_ZN11SupercellIdD2Ev)
    RESOLVE(_ZN15SupercellSocialC1EP23SupercellSocialDelegate)
    RESOLVE(_ZN15SupercellSocialC2EP23SupercellSocialDelegate)
    RESOLVE(_ZN15SupercellSocialD0Ev)
    RESOLVE(_ZN15SupercellSocialD1Ev)
    RESOLVE(_ZN15SupercellSocialD2Ev)
#undef RESOLVE

    return true;
}

// Exported JNI entry point
EXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    LOGI("NullsProbe proxy JNI_OnLoad called!");
    resolve_real_symbols();
    jint real_result = JNI_VERSION_1_6;
    if (g_real_scid_handle) {
        typedef jint (*jni_onload_fn)(JavaVM *, void *);
        jni_onload_fn real_onload = reinterpret_cast<jni_onload_fn>(dlsym(g_real_scid_handle, "JNI_OnLoad"));
        if (real_onload) {
            LOGI("Forwarding to real libscid_sdk JNI_OnLoad...");
            real_result = real_onload(vm, reserved);
        }
    }

    // The real SDK must finish its loader-time initialization before the
    // probe scans libg or patches any game instructions.  Starting the worker
    // before this point races the native bridge on MuMu/libndk_translation.
    bool expected = false;
    if (!g_init_worker_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
        LOGI("NullsProbe init worker already scheduled; ignoring duplicate JNI_OnLoad");
        return real_result;
    }
    pthread_t init_thread;
    const int result = pthread_create(&init_thread, nullptr, init_worker_thread, nullptr);
    if (result != 0) {
        LOGE("pthread_create(init worker) failed: %d", result);
    } else {
        pthread_detach(init_thread);
    }
    return real_result;
}
