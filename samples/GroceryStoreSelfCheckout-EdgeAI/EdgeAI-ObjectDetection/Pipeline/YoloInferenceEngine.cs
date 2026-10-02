using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;

namespace EdgeAI_ObjectDetection.Pipeline;

public sealed class YoloInferenceEngine : IDisposable
{
    private const float MinimumDetectionConfidence = 0.5f;
    private static readonly string[] CocoLabels =
    [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ];

    private readonly InferenceSession _session;

    public string ExecutionProvider { get; }
    public string HardwareDevice { get; }

    private YoloInferenceEngine(InferenceSession session, string executionProvider, string hardwareDevice)
    {
        _session = session;
        ExecutionProvider = executionProvider;
        HardwareDevice = hardwareDevice;
    }

    public static YoloInferenceEngine Create(string modelPath, OrtEpDevice selectedDevice)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);

        return new YoloInferenceEngine(
            CreateSessionForDevice(modelPath, selectedDevice),
            selectedDevice.EpName,
            selectedDevice.HardwareDevice.Type.ToString());
    }

    public IReadOnlyList<YoloDetection> Run(YoloModelInput input)
    {
        string inputName = _session.InputMetadata.Keys.FirstOrDefault()
            ?? throw new InvalidOperationException("The selected model has no input tensor.");
        DenseTensor<float> tensor = new(input.Tensor, [1, 3, YoloPreprocessor.InputSize, YoloPreprocessor.InputSize]);
        NamedOnnxValue modelInput = NamedOnnxValue.CreateFromTensor(inputName, tensor);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _session.Run([modelInput]);
        DisposableNamedOnnxValue output = outputs.FirstOrDefault()
            ?? throw new InvalidOperationException("The selected model returned no output tensors.");
        return ParseDetections(output.AsTensor<float>());
    }

    private static IReadOnlyList<YoloDetection> ParseDetections(Tensor<float> output)
    {
        if (output.Rank != 3 || output.Dimensions[0] != 1 || output.Dimensions[2] != 6)
            throw new InvalidOperationException(
                "Expected YOLO26 NMS output shaped [1, detections, 6] (x1, y1, x2, y2, confidence, COCO class id).");

        List<YoloDetection> detections = new();
        for (int index = 0; index < output.Dimensions[1]; index++)
        {
            float confidence = output[0, index, 4];
            float classIdValue = output[0, index, 5];
            if (!float.IsFinite(confidence) || confidence < MinimumDetectionConfidence ||
                !TryGetCocoLabel(classIdValue, out string label))
                continue;

            float x1 = output[0, index, 0];
            float y1 = output[0, index, 1];
            float x2 = output[0, index, 2];
            float y2 = output[0, index, 3];
            if (!float.IsFinite(x1) || !float.IsFinite(y1) ||
                !float.IsFinite(x2) || !float.IsFinite(y2))
                continue;

            x1 = Math.Clamp(x1, 0, YoloPreprocessor.InputSize);
            y1 = Math.Clamp(y1, 0, YoloPreprocessor.InputSize);
            x2 = Math.Clamp(x2, 0, YoloPreprocessor.InputSize);
            y2 = Math.Clamp(y2, 0, YoloPreprocessor.InputSize);
            if (x2 <= x1 || y2 <= y1)
                continue;

            detections.Add(new YoloDetection(label, confidence, x1, y1, x2 - x1, y2 - y1));
        }

        return detections;
    }

    private static bool TryGetCocoLabel(float classIdValue, out string label)
    {
        if (!float.IsFinite(classIdValue))
        {
            label = string.Empty;
            return false;
        }

        int classId = (int)classIdValue;
        if (classIdValue != classId || classId < 0 || classId >= CocoLabels.Length)
        {
            label = string.Empty;
            return false;
        }

        label = CocoLabels[classId];
        return true;
    }

    private static InferenceSession CreateSessionForDevice(string modelPath, OrtEpDevice device)
    {
        using SessionOptions options = new();
        options.AppendExecutionProvider(OrtEnv.Instance(), [device], new Dictionary<string, string>());
        return new InferenceSession(modelPath, options);
    }

    public void Dispose() => _session.Dispose();
}
