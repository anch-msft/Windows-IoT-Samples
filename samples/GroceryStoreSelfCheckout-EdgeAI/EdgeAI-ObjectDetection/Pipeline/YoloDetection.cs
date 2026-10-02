namespace EdgeAI_ObjectDetection.Pipeline;

public sealed record YoloDetection(
    string Label,
    float Confidence,
    float X,
    float Y,
    float Width,
    float Height);
