using Microsoft.ML.OnnxRuntime;
using System;
using Windows.Media.Capture.Frames;

namespace EdgeAI_ObjectDetection.Controls;

public enum InferenceType
{
    Stream,
    OneShot
}
public enum YoloInferenceState
{
    Stopped,
    Starting,
    Running,
    Stopping,
    Disposed
}

public sealed record InferenceSettings
{
    public required MediaFrameSourceGroup FrameSourceGroup
    {
        get;
        init;
    }
    public required string ModelPath
    {
        get;
        init;
    }
    public required OrtEpDevice ExecutionProvider
    {
        get;
        init;
    }
    public double MaxEndToEndFps
    {
        get;
        init;
    } = 30;
    public InferenceType InferenceType
    {
        get;
        init;
    } = InferenceType.Stream;

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(FrameSourceGroup);
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath);
        ArgumentNullException.ThrowIfNull(ExecutionProvider);
        if (!double.IsFinite(MaxEndToEndFps) || MaxEndToEndFps < 0.1 || MaxEndToEndFps > 240)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxEndToEndFps), "Use a rate between 0.1 and 240 FPS.");
        }
        if (!Enum.IsDefined(InferenceType))
        {
            throw new ArgumentOutOfRangeException(nameof(InferenceType));
        }
    }
}
