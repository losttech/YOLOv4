namespace tensorflow.data {
    using System;
    using System.Collections.Generic;
    using System.Drawing;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading;

    using numpy;

    using Python.Runtime;

    using tensorflow.image;

    public class ObjectDetectionDataset {
        readonly string[] annotations;
        readonly string[] classNames;
        readonly uint[] strides;
        readonly ndarray<float> anchors;
        readonly uint anchorsPerScale;
        readonly uint inputSize;
        readonly uint maxBBoxPerScale;

        public ReadOnlySpan<string> ClassNames => this.classNames.AsSpan();
        uint ClassCount => (uint)this.classNames.Length;
        public int Count => this.annotations.Length;

        public ObjectDetectionDataset(string[] annotations, string[] classNames,
                                      int[] strides, int inputSize,
                                      ndarray<float> anchors,
                                      int anchorsPerScale,
                                      int maxBBoxPerScale) {
            this.classNames = classNames ?? throw new ArgumentNullException(nameof(classNames));
            if (classNames.Length == 0)
                throw new ArgumentException(message: "List of class names must not be empty");

            this.annotations = annotations ?? throw new ArgumentNullException(nameof(annotations));
            if (annotations.Length == 0)
                throw new ArgumentException(message: "List of annotations must not be empty");

            if (strides is null || strides.Length == 0)
                throw new ArgumentNullException(nameof(strides));
            if (strides.Any(NotPositive)) throw new ArgumentOutOfRangeException(nameof(strides));
            this.strides = strides.Select(i => (uint)i).ToArray();

            if (anchors is null) throw new ArgumentNullException(nameof(anchors));
            if (anchors.ndim != 3) throw new ArgumentException("Bad shape", paramName: nameof(anchors));
            this.anchors = anchors;

            if (anchorsPerScale <= 0)
                throw new ArgumentOutOfRangeException(nameof(anchorsPerScale));
            this.anchorsPerScale = (uint)anchorsPerScale;

            if (inputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            this.inputSize = (uint)inputSize;

            if (maxBBoxPerScale <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxBBoxPerScale));
            this.maxBBoxPerScale = (uint)maxBBoxPerScale;
        }

        public void Shuffle() => Tools.Shuffle(this.annotations);

        public IEnumerable<EntryBatch> Batch(int batchSize,
                                             Func<Entry<float>, Entry<float>>? onloadAugmentation) {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            uint ubatchSize = (uint)batchSize;

            int totalBatches = (int)Math.Ceiling(this.Count * 1F / batchSize);
            uint[] outputSizes = this.strides.Select(stride => this.inputSize / stride).ToArray();
            for (int batchNo = 0; batchNo < totalBatches; batchNo++) {
                //tf does not seem to be use here
                //using var _ = tf.device("/cpu:0").StartUsing();
                var batchImages = np.zeros<float>(new ulong[] { ubatchSize, this.inputSize, this.inputSize, 3 });
                var batchBBoxLabels = outputSizes.Select(outputSize
                    => np.zeros<float>(new ulong[] {
                                       ubatchSize, outputSize, outputSize,
                                       this.anchorsPerScale, 5 + this.ClassCount })
                    ).ToArray();

                var batchBBoxes = outputSizes.Select(
                        _ => np.zeros<float>(new ulong[] { ubatchSize, this.maxBBoxPerScale, 4 }))
                    .ToArray();

                for (int itemNo = 0; itemNo < batchSize; itemNo++) {
                    int index = totalBatches * batchSize + itemNo;
                    // loop the last few items for the last batch if necessary
                    if (index >= this.Count) index -= this.Count;
                    string annotation = this.annotations[index];
                    var rawEntry = LoadAnnotation(annotation);
                    var entry = new Entry<float> {
                        Image = (ndarray<float>)rawEntry.Image.astype(dtype.GetClass<float>()),
                        BoundingBoxes = rawEntry.BoundingBoxes,
                    };
                    if (onloadAugmentation != null)
                        entry = onloadAugmentation(entry);
                    entry = Preprocess(entry, new Size((int)this.inputSize, (int)this.inputSize));

                    var (labes, boxes) = this.PreprocessTrueBoxes(entry.BoundingBoxes, outputSizes);
                    batchImages[itemNo, .., .., ..] = entry.Image;
                    for (int i = 0; i < outputSizes.Length; i++) {
                        batchBBoxLabels[i][itemNo, .., .., .., ..] = labes[i];
                        batchBBoxes[i][itemNo, .., ..] = boxes[i];
                    }
                }

                yield return new EntryBatch {
                    Images = batchImages,
                    BBoxLabels = batchBBoxLabels,
                    BBoxes = batchBBoxes
                };
            }
        }

        public static string[] LoadAnnotations(TextReader reader) {
            var result = new List<string>();
            for (string line = reader.ReadLine(); line != null; line = reader.ReadLine()) {
                string trimmed = line.Trim();
                if (trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Length > 1)
                    result.Add(trimmed);
            }

            return result.ToArray();
        }

        // TODO: for reproducibility, use numpy.random
        static readonly ThreadLocal<Random> random = new ThreadLocal<Random>(() => new Random());
        public static Entry<T> RandomHorizontalFlip<T>(Entry<T> entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            int width = entry.Image.shape.Item2;
            int[] reversedXs = Enumerable.Range(0, width).Reverse().ToArray();
            entry.Image = (ndarray<T>)entry.Image.__getitem__((.., reversedXs, ..));
            entry.BoundingBoxes[(.., new[] { 0, 2 })] =
                width - entry.BoundingBoxes[(.., new[] { 2, 0 })];

            return entry;
        }

        public static Entry<T> RandomCrop<T>(Entry<T> entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            int h = entry.Image.shape.Item1, w = entry.Image.shape.Item2;
            ndarray<int> maxBBox = np.concatenate(new[] {
                (ndarray<int>)entry.BoundingBoxes[.., 0..2].min(axis: 0),
                (ndarray<int>)entry.BoundingBoxes[.., 2..4].max(axis: 0),
            }, axis: -1);

            int maxLtrans = maxBBox[0].AsScalar();
            int maxUtrans = maxBBox[1].AsScalar();
            int maxRtrans = w - maxBBox[2].AsScalar();
            int maxDtrans = h - maxBBox[3].AsScalar();

            int cropXMin = Math.Max(0, maxBBox[0].AsScalar() - random.Value.Next(maxLtrans));
            int cropYMin = Math.Max(0, maxBBox[1].AsScalar() - random.Value.Next(maxUtrans));
            int cropXMax = Math.Min(w, maxBBox[2].AsScalar() + random.Value.Next(maxRtrans));
            int cropYMax = Math.Min(h, maxBBox[3].AsScalar() + random.Value.Next(maxDtrans));

            entry.Image = entry.Image[cropYMin..cropYMax, cropXMin..cropXMax];
            entry.BoundingBoxes[(.., new[] { 0, 2 })] -= cropXMin;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] -= cropYMin;

            return entry;
        }

        public static Entry<T> RandomTranslate<T>(Entry<T> entry) where T : unmanaged {
            if (random.Value.Next(2) == 0)
                return entry;

            int h = entry.Image.shape.Item1, w = entry.Image.shape.Item2;
            ndarray<int> maxBBox = np.concatenate(new[] {
                (ndarray<int>)entry.BoundingBoxes[.., 0..2].min(axis: 0),
                (ndarray<int>)entry.BoundingBoxes[.., 2..4].max(axis: 0),
            }, axis: -1);

            int maxLtrans = maxBBox[0].AsScalar();
            int maxUtrans = maxBBox[1].AsScalar();
            int maxRtrans = w - maxBBox[2].AsScalar();
            int maxDtrans = h - maxBBox[3].AsScalar();

            // TODO: use numpy.random.uniform for reproducibility?
            var (min, max) = Sort(-(maxLtrans - 1), maxRtrans - 1);
            int tx = random.Value.Next(minValue: min, maxValue: max);
            (min, max) = Sort(-(maxUtrans - 1), maxDtrans - 1);
            int ty = random.Value.Next(minValue: min, maxValue: max);
            entry.Image = TranslateImage(entry.Image, tx: tx, ty: ty);
            entry.BoundingBoxes[(.., new[] { 0, 2 })] += tx;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] += ty;

            return entry;
        }

        static (int, int) Sort(int a, int b) => (Math.Min(a, b), Math.Max(a, b));

        static ndarray<T> TranslateImage<T>(ndarray<T> image, int tx, int ty) where T : unmanaged {
            if (tx == 0 && ty == 0) return image;

            int h = image.shape.Item1, w = image.shape.Item2, c = image.shape.Item3;
            int toX = tx < 0 ? 0 : tx;
            int toY = ty < 0 ? 0 : ty;

            var temp = np.zeros<T>(h + Math.Abs(ty), w + Math.Abs(tx), c);
            temp[toY..(toY + h), toX..(toX + w), ..] = image;

            int fromX = tx < 0 ? -tx : 0;
            int fromY = ty < 0 ? -ty : 0;
            return temp[fromY..(fromY + h), fromX..(fromX + w), ..];
        }

        public static Entry<byte> LoadAnnotation(string annotation) {
            if (annotation is null) throw new ArgumentNullException(nameof(annotation));

            string[] line = annotation.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            string imagePath = line[0];
            ndarray<byte> image = ImageTools.LoadRGB8(imagePath).ToNumPyArray();
            var bboxes = (ndarray<int>)np.array(line.Slice(1..).Select(box =>
                box
                    .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(s => int.Parse(s, CultureInfo.InvariantCulture))
                    .ToArray()));

            return new Entry<byte> {
                Image = image,
                BoundingBoxes = bboxes,
            };
        }

        public static Entry<T> RandomlyApplyAugmentations<T>(Entry<T> entry) where T : unmanaged {
            entry = RandomHorizontalFlip(entry);
            entry = RandomCrop(entry);
            entry = RandomTranslate(entry);
            return entry;
        }

        public static Entry<float> Preprocess(Entry<float> entry, Size targetSize)
            => ImageTools.YoloPreprocess(entry, targetSize);

        static readonly Python.Runtime.PyObject ellipsis;
        static ndarray<float> BBoxIOU(ndarray<float> boxes1, ndarray<float> boxes2) {
            var area1 = boxes1[(ellipsis, 2)] * boxes1[(ellipsis, 3)];
            var area2 = boxes1[(ellipsis, 2)] * boxes1[(ellipsis, 3)];

            boxes1 = np.concatenate(new[] {
                boxes1[(ellipsis, ..2)] - boxes1[(ellipsis, 2..)] * 0.5f,
                boxes1[(ellipsis, ..2)] + boxes1[(ellipsis, 2..)] * 0.5f,
            }, axis: -1);
            boxes2 = np.concatenate(new[] {
                boxes2[(ellipsis, ..2)] - boxes2[(ellipsis, 2..)]*0.5f,
                boxes2[(ellipsis, ..2)] + boxes2[(ellipsis, 2..)]*0.5f,
            }, axis: -1);

            var leftUp = np.maximum(boxes1[(ellipsis, ..2)].AsArray(), boxes2[(ellipsis, ..2)].AsArray());
            var rightDown = np.minimum(boxes1[(ellipsis, 2..)].AsArray(), boxes2[(ellipsis, 2..)].AsArray());

            var intersection = np.maximum(rightDown - leftUp, 0.0f);
            var intersectionArea = intersection[(ellipsis, 0)] * intersection[(ellipsis, 1)];
            var unionArea = area1 + area2 - intersectionArea;

            return (intersectionArea / unionArea).AsArray();
        }
        (ndarray<float>[], ndarray<float>[]) PreprocessTrueBoxes(ndarray<int> bboxes, uint[] outputSizes) {
            var label = outputSizes
                .Select(size => np.zeros<float>(new ulong[] {
                        size, size, this.anchorsPerScale, 5 + this.ClassCount }))
                .ToArray();
            var bboxesXYWH = outputSizes
                .Select(_ => np.zeros<float>(new ulong[] { this.maxBBoxPerScale, 4 }))
                .ToArray();
            var bboxCount = np.zeros<int>((ulong)outputSizes.Length);
            var stridesPlus = np.array(this.strides)[(.., np.newaxis)].AsArray<int>().AsType<float>();

            foreach (ndarray<int> bbox in bboxes) {
                var coords = bbox[..4];
                var classIndex = bbox[4];

                var oneHot = np.zeros<float>(this.ClassCount);
                oneHot[classIndex] = 1;

                var uniform = np.full((int)this.ClassCount,
                        fill_value: 1.0f / this.ClassCount,
                        dtype: dtype.GetClass<float>())
                    .AsArray<float>();
                const float deta = 0.01f;
                var smoothOneHot = oneHot * (1 - deta) + deta * uniform;

                var bboxXYWH = np.concatenate(new[] {
                    (coords[2..] + coords[..2]).AsType<float>() * 0.5f,
                    (coords[2..] - coords[..2]).AsType<float>()
                }, axis: -1);
                var bboxXYWHScaled = (1.0f * bboxXYWH[(np.newaxis, ..)] / stridesPlus).AsArray();

                var iou = new List<object>();

                void UpdateBoxesAtScale(int scale, object iouMaskOrIndex) {
                    var indices = bboxXYWHScaled[scale, 0..2].AsType<int>();
                    ArrayOrElement<int> xind = indices[0], yind = indices[1];

                    label[scale][(yind, xind, iouMaskOrIndex, ..)] = 0;
                    label[scale][(yind, xind, iouMaskOrIndex, 0..4)] = bboxXYWH;
                    label[scale][(yind, xind, iouMaskOrIndex, 4..5)] = 1.0f;
                    label[scale][(yind, xind, iouMaskOrIndex, 5..)] = smoothOneHot;

                    int bboxIndex = bboxCount[scale].AsScalar() % (int)this.maxBBoxPerScale;
                    bboxesXYWH[scale][bboxIndex, ..4] = bboxXYWH;
                    bboxCount[scale] += 1;
                }

                bool positiveExists = false;
                for (int scaleIndex = 0; scaleIndex < outputSizes.Length; scaleIndex++) {
                    uint outputSize = outputSizes[scaleIndex];
                    var anchorsXYWH = np.zeros<float>(this.anchorsPerScale, 4);
                    anchorsXYWH[.., 0..2] = anchorsXYWH[.., 0..2].AsType<int>().AsType<float>() + 0.5f;
                    anchorsXYWH[.., 2..4] = this.anchors[scaleIndex].AsArray();

                    var iouScale = BBoxIOU(bboxXYWHScaled[scaleIndex][(np.newaxis, ..)].AsArray(),
                                           anchorsXYWH);
                    iou.Add(iouScale);
                    var iouMask = iouScale > 0.3f;

                    if (iouMask.any()) {
                        UpdateBoxesAtScale(scaleIndex, iouMask);

                        positiveExists = true;
                    }
                }

                if (!positiveExists) {
                    int bestAnchorIndex = np.array(iou).reshape(-1).argmax(axis: -1).AsScalar<int>();
                    int bestDetection = bestAnchorIndex / (int)this.anchorsPerScale;
                    int bestAnchor = bestAnchorIndex % (int)this.anchorsPerScale;

                    UpdateBoxesAtScale(bestDetection, bestAnchor);
                }
            }

            return (label, bboxesXYWH);
        }

        public struct EntryBatch {
            public ndarray<float> Images { get; set; }
            public ndarray<float>[] BBoxLabels { get; set; }
            public ndarray<float>[] BBoxes { get; set; }
        }

        public struct Entry<T> {
            /// <summary>HWC image</summary>
            public ndarray<T> Image { get; set; }
            public ndarray<int> BoundingBoxes { get; set; }
        }

        public static class Entry {
            public static (Range, int[]) AllHorizontal { get; } = (.., new[] { 0, 2 });
            public static (Range, int[]) AllVertical { get; } = (.., new[] { 1, 3 });
        }

        static bool NotPositive(int value) => value <= 0;

        static ObjectDetectionDataset() {
            using var _ = Py.GIL();
            ellipsis = PythonEngine.Eval("...");
        }
    }
}
