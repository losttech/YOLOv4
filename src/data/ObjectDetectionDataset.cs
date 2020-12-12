namespace tensorflow.data {
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Threading;

    using numpy;

    using Python.Runtime;

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.Advanced;
    using SixLabors.ImageSharp.PixelFormats;
    using SixLabors.ImageSharp.Processing;

    using tensorflow.image;

    using Image = SixLabors.ImageSharp.Image;
    using Rectangle = SixLabors.ImageSharp.Rectangle;
    using Size = SixLabors.ImageSharp.Size;

    public class ObjectDetectionDataset {
        readonly string[] annotations;
        readonly string[] classNames;
        readonly int[] strides;
        readonly ndarray<float> anchors;
        readonly int anchorsPerScale;
        readonly int inputSize;
        readonly int maxBBoxPerScale;

        public int InputSize => (int)this.inputSize;
        public ReadOnlySpan<string> ClassNames => this.classNames;
        public ReadOnlySpan<int> Strides => this.strides;
        int ClassCount => this.classNames.Length;
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
            this.strides = strides.ToArray();

            if (anchors is null) throw new ArgumentNullException(nameof(anchors));
            if (anchors.ndim != 3) throw new ArgumentException("Bad shape", paramName: nameof(anchors));
            this.anchors = anchors;

            if (anchorsPerScale <= 0)
                throw new ArgumentOutOfRangeException(nameof(anchorsPerScale));
            this.anchorsPerScale = anchorsPerScale;

            if (inputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            this.inputSize = inputSize;

            if (maxBBoxPerScale <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxBBoxPerScale));
            this.maxBBoxPerScale = maxBBoxPerScale;
        }

        public void Shuffle() => Tools.Shuffle(this.annotations);

        public IReadOnlyList<EntryBatch> Batch(int batchSize,
                                             Func<ClrEntry, ClrEntry>? onloadAugmentation) {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

            return new BatchList(this, batchSize: batchSize, onloadAugmentation);
        }

        class BatchList: IReadOnlyList<EntryBatch> {
            readonly ObjectDetectionDataset dataset;
            public Func<ClrEntry, ClrEntry>? OnloadAugmentation { get; }
            public int BatchSize { get; }
            public int Count { get; }

            public EntryBatch this[int index] => this.dataset.GetBatch(this.BatchSize, index, this.OnloadAugmentation);

            public BatchList(ObjectDetectionDataset dataset, int batchSize, Func<ClrEntry, ClrEntry>? onloadAugmentation) {
                if (dataset is null) throw new ArgumentNullException(nameof(dataset));
                if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
                this.dataset = dataset;
                this.BatchSize = batchSize;
                this.Count = this.dataset.BatchCount(this.BatchSize);
                this.OnloadAugmentation = onloadAugmentation;
            }

            public IEnumerator<EntryBatch> GetEnumerator() {
                for(int batch = 0; batch < this.Count; batch++)
                    yield return this[batch];
            }
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => this.GetEnumerator();
        }

        public int BatchCount(int batchSize) {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            return (int)Math.Ceiling(this.Count * 1F / batchSize);
        }

        public EntryBatch GetBatch(int batchSize, int batchIndex,
                                   Func<ClrEntry, ClrEntry>? onloadAugmentation) {
            if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
            int totalBatches = this.BatchCount(batchSize);
            if (batchIndex < 0 || batchIndex >= totalBatches)
                throw new IndexOutOfRangeException();

            int[] outputSizes = this.strides.Select(stride => this.inputSize / stride).ToArray();

            var batchImages = np.zeros<float>(batchSize, this.inputSize, this.inputSize, 3);
            var batchBBoxLabels = outputSizes.Select(outputSize
                => np.zeros<float>(
                                   batchSize, outputSize, outputSize,
                                   this.anchorsPerScale, 5 + this.ClassCount)
                ).ToArray();

            var batchBBoxes = outputSizes.Select(
                    _ => np.zeros<float>(batchSize, this.maxBBoxPerScale, 4))
                .ToArray();

            for (int itemNo = 0; itemNo < batchSize; itemNo++) {
                int index = batchIndex * batchSize + itemNo;
                // loop the last few items for the last batch if necessary
                if (index >= this.Count) index -= this.Count;
                string annotation = this.annotations[index];
                var rawEntry = LoadAnnotationClr(annotation);
                if (onloadAugmentation != null)
                    rawEntry = onloadAugmentation(rawEntry);
                var entry = Preprocess(rawEntry, new Size(this.inputSize, this.inputSize));

                var (labels, boxes) = this.PreprocessTrueBoxes(entry.BoundingBoxes, outputSizes);

                batchImages[itemNo, .., .., ..] = entry.Image;
                for (int i = 0; i < outputSizes.Length; i++) {
                    batchBBoxLabels[i][itemNo, .., .., .., ..] = labels[i];
                    batchBBoxes[i][itemNo, .., ..] = boxes[i];
                }
            }

            return new EntryBatch {
                Images = batchImages,
                BBoxLabels = batchBBoxLabels,
                BBoxes = batchBBoxes
            };
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
            entry.Image = (ndarray<T>)entry.Image[.., reversedXs, ..];
            entry.BoundingBoxes[.., new[] { 0, 2 }] =
                width - entry.BoundingBoxes[.., new[] { 2, 0 }];

            return entry;
        }
        public static ClrEntry RandomHorizontalFlip(ClrEntry entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            entry.Image.Mutate(x => x.Flip(FlipMode.Horizontal));
            entry.BoundingBoxes[.., new[] { 0, 2 }] = entry.Image.Width - entry.BoundingBoxes[.., new[] { 2, 0 }];

            return entry;
        }

        public static Entry<T> RandomCrop<T>(Entry<T> entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            int h = entry.Image.shape.Item1, w = entry.Image.shape.Item2;
            GetRandomCrop(entry.BoundingBoxes, h, w,
                out int cropXMin, out int cropYMin, out int cropXMax, out int cropYMax);

            entry.Image = entry.Image[cropYMin..cropYMax, cropXMin..cropXMax];
            entry.BoundingBoxes[(.., new[] { 0, 2 })] -= cropXMin;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] -= cropYMin;

            return entry;
        }

        public static ClrEntry RandomCrop(ClrEntry entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            GetRandomCrop(entry.BoundingBoxes, height: entry.Image.Height, width: entry.Image.Width,
                out int cropXMin, out int cropYMin, out int cropXMax, out int cropYMax);

            var rect = new Rectangle(cropXMin, cropYMin, cropXMax - cropXMin, cropYMax - cropYMin);

            entry.Image.Mutate(x => x.Crop(rect));
            entry.BoundingBoxes[(.., new[] { 0, 2 })] -= cropXMin;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] -= cropYMin;

            return entry;
        }

        static void GetRandomCrop(ndarray<int> boundingBoxes, int height, int width,
                                  out int cropXMin, out int cropYMin, out int cropXMax, out int cropYMax) {
            ndarray<int> maxBBox = np.concatenate(new[] {
                (ndarray<int>)boundingBoxes[.., 0..2].min(axis: 0),
                (ndarray<int>)boundingBoxes[.., 2..4].max(axis: 0),
            }, axis: -1);

            int maxLtrans = maxBBox[0].AsScalar();
            int maxUtrans = maxBBox[1].AsScalar();
            int maxRtrans = width - maxBBox[2].AsScalar();
            int maxDtrans = height - maxBBox[3].AsScalar();

            cropXMin = Math.Max(0, maxBBox[0].AsScalar() - random.Value.Next(maxLtrans));
            cropYMin = Math.Max(0, maxBBox[1].AsScalar() - random.Value.Next(maxUtrans));
            cropXMax = Math.Min(width, maxBBox[2].AsScalar() + random.Value.Next(maxRtrans));
            cropYMax = Math.Min(height, maxBBox[3].AsScalar() + random.Value.Next(maxDtrans));
        }

        public static Entry<T> RandomTranslate<T>(Entry<T> entry) where T : unmanaged {
            if (random.Value.Next(2) == 0)
                return entry;

            int h = entry.Image.shape.Item1, w = entry.Image.shape.Item2;
            GetRandomTranslation(entry.BoundingBoxes, h, w, out int tx, out int ty);

            entry.Image = TranslateImage(entry.Image, tx: tx, ty: ty);
            entry.BoundingBoxes[(.., new[] { 0, 2 })] += tx;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] += ty;

            return entry;
        }

        public static ClrEntry RandomTranslate(ClrEntry entry) {
            if (random.Value.Next(2) == 0)
                return entry;

            GetRandomTranslation(entry.BoundingBoxes,
                                 height: entry.Image.Height, width: entry.Image.Width,
                                 out int tx, out int ty);

            if (tx == 0 && ty == 0) return entry;

            var rect = new Rectangle(-tx, -ty, entry.Image.Width, entry.Image.Height);
            var translated = new Image<Rgb24>(entry.Image.Width, entry.Image.Height, Color.Black);
            translated.Mutate(x => x.DrawImage(entry.Image, new Point(tx, ty), opacity: 1));
            entry.Image = translated;
            entry.BoundingBoxes[(.., new[] { 0, 2 })] += tx;
            entry.BoundingBoxes[(.., new[] { 1, 3 })] += ty;

            return entry;
        }

        static void GetRandomTranslation(ndarray<int> boundingBoxes, int height, int width,
                                            out int tx, out int ty) {
            ndarray<int> maxBBox = np.concatenate(new[] {
                (ndarray<int>)boundingBoxes[.., 0..2].min(axis: 0),
                (ndarray<int>)boundingBoxes[.., 2..4].max(axis: 0),
            }, axis: -1);

            int maxLtrans = maxBBox[0].AsScalar();
            int maxUtrans = maxBBox[1].AsScalar();
            int maxRtrans = width - maxBBox[2].AsScalar();
            int maxDtrans = height - maxBBox[3].AsScalar();

            // TODO: use numpy.random.uniform for reproducibility?
            var (min, max) = Sort(-(maxLtrans - 1), maxRtrans - 1);
            tx = random.Value.Next(minValue: min, maxValue: max);
            (min, max) = Sort(-(maxUtrans - 1), maxDtrans - 1);
            ty = random.Value.Next(minValue: min, maxValue: max);
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
            ndarray<int> bboxes = LoadBBoxes(line.Slice(1..));

            return new Entry<byte> {
                Image = image,
                BoundingBoxes = bboxes,
            };
        }

        public static ClrEntry LoadAnnotationClr(string annotation) {
            if (annotation is null) throw new ArgumentNullException(nameof(annotation));

            string[] line = annotation.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            string imagePath = line[0];
            var image = Image.Load<Rgb24>(imagePath);
            ndarray<int> bboxes = LoadBBoxes(line.Slice(1..));

            return new ClrEntry {
                Image = image,
                BoundingBoxes = bboxes,
            };
        }

        static ndarray<int> LoadBBoxes(string[] bboxTexts)
            => (ndarray<int>)np.array(bboxTexts
                .Select(box =>box
                             .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                             .Select(s => int.Parse(s, CultureInfo.InvariantCulture))
                             .ToArray()));

        public static Entry<T> RandomlyApplyAugmentations<T>(Entry<T> entry) where T : unmanaged {
            entry = RandomHorizontalFlip(entry);
            entry = RandomCrop(entry);
            entry = RandomTranslate(entry);
            return entry;
        }

        public static ClrEntry RandomlyApplyAugmentations(ClrEntry entry) {
            entry = RandomHorizontalFlip(entry);
            entry = RandomCrop(entry);
            entry = RandomTranslate(entry);
            return entry;
        }

        public static Entry<float> Preprocess(ClrEntry entry, Size targetSize)
            => ImageTools.YoloPreprocess(entry, targetSize);

        static readonly PyObject ellipsis;
        internal static ndarray<float> BBoxIOU(ndarray<float> boxes1, ndarray<float> boxes2) {
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
            var epsilon = new float32(tf.keras.backend.epsilon());
            var unionArea = np.maximum(area1 + area2 - intersectionArea, epsilon);

            return np.maximum(epsilon, (intersectionArea / unionArea).AsArray<float>());
        }
        (ndarray<float>[], ndarray<float>[]) PreprocessTrueBoxes(ndarray<int> bboxes, int[] outputSizes) {
            var label = outputSizes
                .Select(size => np.zeros<float>(
                        size, size, this.anchorsPerScale, 5 + this.ClassCount))
                .ToArray();
            var bboxesXYWH = outputSizes
                .Select(_ => np.zeros<float>(this.maxBBoxPerScale, 4))
                .ToArray();
            var bboxCount = np.zeros<int>(outputSizes.Length);
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

                var iou = new List<ndarray<float>>();

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
                    int outputSize = outputSizes[scaleIndex];
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
                    int bestAnchorIndex = (int)np.array(iou).reshape(-1).argmax(axis: -1).AsScalar<long>();
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

        public struct ClrEntry {
            public Image<Rgb24> Image { get; set; }
            public ndarray<int> BoundingBoxes { get; set; }

            public Entry<float> ToNumPyEntry() {
                var numpyImage = np.zeros<byte>(this.Image.Height, this.Image.Width * 3);
                for(int y = 0; y < this.Image.Height; y++) {
                    var row = this.Image.GetPixelRowMemory(y);
                    numpyImage[y] = MarshalingExtensions.ToNumPyArray<byte>(
                        MemoryMarshal.Cast<Rgb24, byte>(row.Span));
                }
                return new Entry<float> {
                    Image = ((ndarray<byte>)numpyImage.reshape(new[] { this.Image.Height, this.Image.Width, 3 }))
                            .AsType<float>(),
                    BoundingBoxes = this.BoundingBoxes,
                };
            }

            public static ClrEntry FromNumPyEntry(Entry<float> entry) {
                int height = entry.Image.shape.Item1;
                int width = entry.Image.shape.Item2;
                var image = new Image<Rgb24>(width: width, height: height);
                var bytes = entry.Image.reshape(new[] { height, width * 3 });
                for (int y = 0; y < height; y++) {
                    var row = MemoryMarshal.Cast<Rgb24, byte>(image.GetPixelRowMemory(y).Span);
                    var byteRow = bytes[y];
                    for (int byteOffset = 0; byteOffset < width * 3; byteOffset++)
                        row[byteOffset] = (byte)(byteRow[byteOffset].AsScalar<float>() * 255);
                }
                return new ClrEntry { Image = image, BoundingBoxes = entry.BoundingBoxes };
            }
        }

        public static class Entry {
            public static (Range, int[]) AllHorizontal { get; } = (.., new[] { 0, 2 });
            public static (Range, int[]) AllVertical { get; } = (.., new[] { 1, 3 });
        }

        static bool NotPositive(int value) => value <= 0;

        public static ndarray<float> ParseAnchors(string anchors)
            => anchors.Split(',')
                .Select(coord => float.Parse(coord.Trim(), CultureInfo.InvariantCulture))
                .ToNumPyArray()
                .reshape(new[] { 3, 3, 2 })
                .AsArray<float>();

        public static ndarray<float> ParseAnchors(IEnumerable<int> anchors)
            => anchors.ToNumPyArray()
                .reshape(new[] { 3, 3, 2 })
                .AsArray<int>()
                .AsType<float>();

        static ObjectDetectionDataset() {
            using var _ = Py.GIL();
            ellipsis = PythonEngine.Eval("...");
        }
    }
}
