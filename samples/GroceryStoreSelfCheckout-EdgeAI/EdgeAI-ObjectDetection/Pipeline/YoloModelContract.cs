using System;
using System.Collections.Generic;
using System.Linq;

namespace EdgeAI_ObjectDetection.Pipeline;

internal static class YoloModelContract
{
    public const int InputSize = 640;
    public const int DetectionLimit = 300;

    public static void ValidateShapes(IEnumerable<int> input, IEnumerable<int> output)
    {
        if (!input.SequenceEqual(new[] { 1, 3, InputSize, InputSize }) ||
            !output.SequenceEqual(new[] { 1, DetectionLimit, 6 }))
        {
            throw new InvalidOperationException(
                "Supported model contract: float32 input [1,3,640,640] and output [1,300,6], processed COCO detections.");
        }
    }
}
