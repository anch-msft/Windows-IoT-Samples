namespace EdgeAI_ObjectDetection.Pipeline;

public sealed record YoloModelInput(
    float[] Tensor,
    int SourceWidth,
    int SourceHeight,
    int CropX,
    int CropY,
    int CropSize);
