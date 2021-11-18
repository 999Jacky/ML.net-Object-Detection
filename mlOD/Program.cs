using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace mlOD {
    class Program {
        private static MLContext mlContext;
        private static ITransformer mlModel;

        // 辨識結果標籤
        public static string[] Labels = new[] { "y", "o", "w3", "w2", "r", "w1" };

        // 辨識結果顏色
        private static Color[] classColors = new Color[] {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange
        };

        private static void Main(string[] args) {
            // 必要檔案路徑
            var assetsRelativePath = @"assets";
            var assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "model", "model.onnx");
            var imagesFolder = Path.Combine(assetsPath, "img");
            var outputFolder = Path.Combine(assetsPath, "output");
            // Directory.CreateDirectory(outputFolder);
            // 計時物件
            var sw = new System.Diagnostics.Stopwatch();
            sw.Reset();
            sw.Start();

            mlContext = new MLContext();
            mlModel = SetupMlnetModel(modelFilePath);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInputData, ImageOutputData>(mlModel);

            // 讀取圖片
            var imgs = ImageInputData.ReadFromFile(imagesFolder);
            sw.Stop();
            var resultLoad = sw.Elapsed.TotalMilliseconds.ToString();
            Console.WriteLine("LOAD Model：" + resultLoad + "ms");

            foreach (var v in imgs) {
                // 辨識
                sw.Reset();
                sw.Start();
                var output = predictionEngine.Predict(v);
                var boxes = ParseResult(output);
                var outImg = DrawBoxesOnBitmap(boxes, v.Img);
                outImg.Save(Path.Combine(outputFolder, v.ImgName), ImageFormat.Jpeg);
                sw.Stop();
                var result = sw.Elapsed.TotalMilliseconds.ToString();
                Console.WriteLine($"{v.ImgName}:{output.scores[0]} ,{result}ms");
            }
        }

        public static string GetAbsolutePath(string relativePath) {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private static ITransformer SetupMlnetModel(string tensorFlowModelFilePath) {
            var pipeline =
                (mlContext.Transforms.ExtractPixels(
                        orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ARGB,
                        colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Rgb,
                        // 必要
                        interleavePixelColors: true,
                        // 設定輸出為byte(uint8)
                        outputAsFloatArray: false,
                        inputColumnName: nameof(ImageInputData.Img),
                        outputColumnName: ModelSettings.inputTensorName)
                    .Append(mlContext.Transforms.ApplyOnnxModel(
                        shapeDictionary: new Dictionary<string, int[]>() {
                            {
                                // 修改餵給模組的輸入格式
                                "image_tensor:0", new[] { 1, 480, 640, 3 }
                            }
                        },
                        modelFile: tensorFlowModelFilePath,
                        outputColumnNames: new[] {
                            ModelSettings.outputTensorNameNum,
                            ModelSettings.outputTensorNameBox,
                            ModelSettings.outputTensorNameScore,
                            ModelSettings.outputTensorNameClass
                        },
                        // inputColumnNames: new[] { ModelSettings.inputTensorName }, gpuDeviceId: 0,
                        inputColumnNames: new[] { ModelSettings.inputTensorName },
                        recursionLimit: 100)));

            ITransformer mlModel = pipeline.Fit(CreateEmptyDataView());

            return mlModel;
        }

        private static IDataView CreateEmptyDataView() {
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(new ImageInputData() {
                Img = new System.Drawing.Bitmap(ImageSettings.imageWidth, ImageSettings.imageHeight)
            });

            var dv = mlContext.Data.LoadFromEnumerable<ImageInputData>(list);
            return dv;
        }

        public static List<BoundingBox> ParseResult(ImageOutputData output) {
            List<BoundingBox> parsed = new List<BoundingBox>();
            for (int i = 0; i < output.num_detection[0]; i++) {
                // 過濾分數過低的辨識結果
                if (output.scores[i] < 0.80) {
                    break;
                }

                BoundingBox b = new BoundingBox();
                b.BoxColor = classColors[(int)output.classes[i] - 1];
                b.Label = Labels[(int)output.classes[i] - 1];
                b.score = output.scores[i];
                float ymin, xmin, ymax, xmax;
                ymin = output.boxes[i * 4] * ImageSettings.imageHeight;
                xmin = output.boxes[(i * 4) + 1] * ImageSettings.imageWidth;
                ymax = output.boxes[(i * 4) + 2] * ImageSettings.imageHeight;
                xmax = output.boxes[(i * 4) + 3] * ImageSettings.imageWidth;

                // float left = xmin * ImageSettings.imageWidth;
                // float right = xmax * ImageSettings.imageWidth;
                // float top = ymin * ImageSettings.imageHeight;
                // float bottom = ymax * ImageSettings.imageHeight;

                b.box = new BoxClass {
                    X = xmin,
                    Y = ymin,
                    Height = ymax - ymin,
                    Width = xmax - xmin
                };
                parsed.Add(b);
            }

            return parsed;
        }

        public static Bitmap DrawBoxesOnBitmap(List<BoundingBox> list, Bitmap img) {
            // 畫出辨識框
            foreach (var box in list) {
                string text = $"{box.Label} ({(box.score * 100).ToString("0")}%)";
                var x = (uint)Math.Max(box.box.X, 0);
                var y = (uint)Math.Max(box.box.Y, 0);
                using (Graphics thumbnailGraphic = Graphics.FromImage(img)) {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);
                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);
                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width,
                        (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, box.Rect);
                }
            }

            return img;
        }

        public struct ModelSettings {
            // input tensor name
            public const string inputTensorName = "image_tensor:0";

            // output tensor name
            public const string outputTensorNameNum = "num_detections:0";
            public const string outputTensorNameBox = "detection_boxes:0";
            public const string outputTensorNameScore = "detection_scores:0";
            public const string outputTensorNameClass = "detection_classes:0";
        }

        public struct ImageSettings {
            public const int imageHeight = 480;
            public const int imageWidth = 640;
        }

        public class ImageInputData {
            // 定義模組輸入
            [ImageType(480, 640)] public Bitmap Img { get; set; }
            public string ImgName { get; set; }

            public static IEnumerable<ImageInputData> ReadFromFile(string imageFolder) {
                return Directory
                    .GetFiles(imageFolder)
                    .Where(filePath => Path.GetExtension(filePath) == ".jpg")
                    .Select(filePath => new ImageInputData() {
                        // 強制指定bitmap的解析度，避免輸出時圖片方向錯誤
                        Img = new Bitmap(new Bitmap(filePath), ImageSettings.imageWidth, ImageSettings.imageHeight),
                        ImgName = Path.GetFileName(filePath)
                    });
            }

            public static Bitmap rotateImg(Bitmap img) {
                img.RotateFlip(rotateFlipType: RotateFlipType.Rotate90FlipNone);
                return img;
            }
        }

        public class ImageOutputData {
            // 定義模組輸出
            [ColumnName(ModelSettings.outputTensorNameNum)]
            public float[] num_detection { get; set; }

            [ColumnName(ModelSettings.outputTensorNameBox)]
            public float[] boxes { get; set; }

            [ColumnName(ModelSettings.outputTensorNameScore)]
            public float[] scores { get; set; }

            [ColumnName(ModelSettings.outputTensorNameClass)]
            public float[] classes { get; set; }
        }

        public class BoxClass {
            public float X { get; set; }
            public float Y { get; set; }
            public float Height { get; set; }
            public float Width { get; set; }
        }

        public class BoundingBox {
            public BoxClass box { get; set; }

            public string Label { get; set; }

            public float score { get; set; }

            // 左上角x,y
            public Rectangle Rect => new Rectangle((int)box.X, (int)box.Y, (int)box.Width,
                (int)box.Height);

            public Color BoxColor { get; set; }
        }
    }
}