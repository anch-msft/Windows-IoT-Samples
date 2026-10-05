using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using EdgeAI_ObjectDetection.Controls;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace EdgeAI_ObjectDetection.Pipeline;

public sealed class YoloInferenceEngine : IDisposable
{
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
    public static IReadOnlyList<string> SupportedClasses
    {
        get;
    } = Array.AsReadOnly(CocoLabels);

    public string ExecutionProvider
    {
        get;
    }
    public string HardwareDevice
    {
        get;
    }

    private YoloInferenceEngine(InferenceSession session, string executionProvider, string hardwareDevice)
    {
        _session = session;
        ExecutionProvider = executionProvider;
        HardwareDevice = hardwareDevice;
    }

    public static YoloInferenceEngine Create(string modelPath, OrtEpDevice selectedDevice)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        InferenceSession session = CreateSessionForDevice(modelPath, selectedDevice);
        try
        {
            ValidateModel(session);
            return new YoloInferenceEngine(session, selectedDevice.EpName, selectedDevice.HardwareDevice.Type.ToString());
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static void ValidateModel(InferenceSession session)
    {
        if (session.InputMetadata.Count != 1 || session.OutputMetadata.Count != 1)
        {
            throw new InvalidOperationException("Expected exactly one YOLO input and one output.");
        }

        NodeMetadata input = session.InputMetadata.Values.Single();
        NodeMetadata output = session.OutputMetadata.Values.Single();
        if (!input.IsTensor || input.ElementType != typeof(float) ||
            !output.IsTensor || output.ElementType != typeof(float))
        {
            throw new InvalidOperationException("Expected float32 input and output tensors.");
        }

        YoloModelContract.ValidateShapes(input.Dimensions, output.Dimensions);
    }

    public IReadOnlyList<Detection> Run(YoloModelInput input)
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

    private static IReadOnlyList<Detection> ParseDetections(Tensor<float> output)
    {
        if (output.Rank != 3 || output.Dimensions[0] != 1 ||
            output.Dimensions[1] != YoloModelContract.DetectionLimit || output.Dimensions[2] != 6)
        {
            throw new InvalidOperationException(
                "Expected processed YOLO output [1,300,6] (x1, y1, x2, y2, confidence, COCO class id).");
        }

        List<Detection> detections = new();
        for (int index = 0; index < output.Dimensions[1]; index++)
        {
            Detection? detection = DetectionDecoder.Decode(
                output[0, index, 0], output[0, index, 1], output[0, index, 2], output[0, index, 3],
                output[0, index, 4], output[0, index, 5], SupportedClasses);
            if (detection is not null)
            {
                detections.Add(detection);
            }
        }

        return detections.AsReadOnly();
    }

    private static InferenceSession CreateSessionForDevice(string modelPath, OrtEpDevice device)
    {
        using SessionOptions options = new();
        options.AppendExecutionProvider(OrtEnv.Instance(), [device], new Dictionary<string, string>());
        return new InferenceSession(modelPath, options);
    }

    public void Dispose() => _session.Dispose();
}
