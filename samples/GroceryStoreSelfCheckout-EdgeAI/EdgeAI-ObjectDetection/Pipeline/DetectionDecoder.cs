using EdgeAI_ObjectDetection.Controls;
using System;
using System.Collections.Generic;

namespace EdgeAI_ObjectDetection.Pipeline;

internal static class DetectionDecoder
{
    public static Detection? Decode(float x1, float y1, float x2, float y2,
        float confidence, float classId, IReadOnlyList<string> labels)
    {
        if (!float.IsFinite(confidence) || confidence < 0.5f || confidence > 1 ||
            !float.IsFinite(classId) || classId < 0 || classId >= labels.Count || classId != (int)classId ||
            !float.IsFinite(x1) || !float.IsFinite(y1) || !float.IsFinite(x2) || !float.IsFinite(y2))
        {
            return null;
        }

        x1 = Math.Clamp(x1 / YoloModelContract.InputSize, 0, 1);
        y1 = Math.Clamp(y1 / YoloModelContract.InputSize, 0, 1);
        x2 = Math.Clamp(x2 / YoloModelContract.InputSize, 0, 1);
        y2 = Math.Clamp(y2 / YoloModelContract.InputSize, 0, 1);
        if (x2 <= x1 || y2 <= y1)
        {
            return null;
        }
        return new Detection((int)classId, labels[(int)classId], confidence,
            new DetectionBox(x1, y1, x2 - x1, y2 - y1));
    }
}
