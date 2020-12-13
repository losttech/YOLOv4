namespace tensorflow.keras.applications {
    using numpy;

    using SixLabors.ImageSharp;

    public class ObjectDetectionResult {
        public int Class { get; set; }
        public float Score { get; set; }
        public RectangleF Box { get; set; }

        public static ObjectDetectionResult[] FromCombinedNonMaxSuppressionBatch(
            ndarray<float> boxes, ndarray<float> scores, ndarray<long> classes,
            int detectionCount) {
            var result = new ObjectDetectionResult[detectionCount];
            for(int detection = 0; detection < detectionCount; detection++) {
                result[detection] = new ObjectDetectionResult {
                    Class = checked((int)classes[0, detection].AsScalar()),
                    Box = ToBox(boxes[0, detection].AsArray()),
                    Score = scores[0, detection].AsScalar(),
                };
            }
            return result;
        }

        static RectangleF ToBox(ndarray<float> tlbr) {
            var (y1, x1, y2, x2) = (tlbr[0].AsScalar(), tlbr[1].AsScalar(), tlbr[2].AsScalar(), tlbr[3].AsScalar());
            return new RectangleF(x: x1, y: y1, width: x2 - x1, height: y2 - y1);
        }
    }
}
