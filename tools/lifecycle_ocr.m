#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
int main(void) { @autoreleasepool {
    NSData *data = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
    VNRecognizeTextRequest *request = [VNRecognizeTextRequest new];
    request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    request.usesLanguageCorrection = NO;
    request.recognitionLanguages = @[@"en-US"];
    NSError *error = nil;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithData:data options:@{}];
    if (![handler performRequests:@[request] error:&error]) { NSLog(@"%@",error); return 1; }
    NSMutableArray *rows = [NSMutableArray new];
    for (VNRecognizedTextObservation *item in request.results) {
        VNRecognizedText *text = [[item topCandidates:1] firstObject];
        CGRect r = item.boundingBox;
        if (text) [rows addObject:@{@"text":text.string, @"confidence":@(text.confidence),
            @"x":@(CGRectGetMidX(r)), @"y":@(1-CGRectGetMidY(r))}];
    }
    NSData *output = [NSJSONSerialization dataWithJSONObject:rows options:0 error:&error];
    [[NSFileHandle fileHandleWithStandardOutput] writeData:output];
    return 0;
} }
