#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

API_AVAILABLE(macos(26.0))
int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [device newCommandAllocator];
        id<MTL4CommandBuffer> cb = [device newCommandBuffer];
        [cb beginCommandBufferWithAllocator:alloc];
        id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
        Class cls = [enc class];
        printf("encoder class: %s (super: %s)\n", class_getName(cls), class_getName(class_getSuperclass(cls)));

        // walk up hierarchy, list methods mentioning interesting keywords
        const char* keys[] = {"timestamp", "Timestamp", "barrier", "Barrier", "split", "Split",
                              "endEncoding", "dispatch", "Dispatch", "sample", "Sample", "counter", "Counter", 0};
        for (Class c = cls; c && c != [NSObject class]; c = class_getSuperclass(c)) {
            unsigned int n = 0;
            Method* methods = class_copyMethodList(c, &n);
            for (unsigned int i = 0; i < n; i++) {
                SEL sel = method_getName(methods[i]);
                const char* name = sel_getName(sel);
                for (int k = 0; keys[k]; k++) {
                    if (strstr(name, keys[k])) {
                        printf("  [%s %s] imp=%p\n", class_getName(c), name, method_getImplementation(methods[i]));
                        break;
                    }
                }
            }
            free(methods);
        }

        MTL4CounterHeapDescriptor* hd = [MTL4CounterHeapDescriptor new];
        hd.type = MTL4CounterHeapTypeTimestamp;
        hd.count = 16;
        id<MTL4CounterHeap> heap = [device newCounterHeapWithDescriptor:hd error:nil];
        printf("counter heap class: %s\n", class_getName([heap class]));

        // dump code bytes of writeTimestampWithGranularity + barrier + endEncoding for offline disasm
        SEL sels[] = {
            @selector(writeTimestampWithGranularity:intoHeap:atIndex:),
            @selector(barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:),
            @selector(dispatchThreads:threadsPerThreadgroup:),
            @selector(endEncoding),
        };
        const char* files[] = {"imp_timestamp.bin", "imp_barrier.bin", "imp_dispatch.bin", "imp_endencoding.bin"};
        for (int i = 0; i < 4; i++) {
            Method m = class_getInstanceMethod(cls, sels[i]);
            if (!m) { printf("missing selector %s\n", sel_getName(sels[i])); continue; }
            void* imp = (void*)method_getImplementation(m);
            FILE* f = fopen(files[i], "wb");
            fwrite(imp, 1, 4096, f);
            fclose(f);
            printf("dumped %s @ %p\n", files[i], imp);
        }
        [enc endEncoding];
        [cb endCommandBuffer];
    }
    return 0;
}
