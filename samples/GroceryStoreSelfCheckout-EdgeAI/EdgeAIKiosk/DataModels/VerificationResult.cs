using EdgeAI_ObjectDetection.Controls;
using System.Collections.Generic;

namespace EdgeAIKiosk.Models;

public record VerificationResult(
    bool IsMatch,
    IReadOnlyList<ScannedItem> ScannedItems,
    IReadOnlyList<Detection> DetectedItems,
    IReadOnlyList<string> Mismatches);
