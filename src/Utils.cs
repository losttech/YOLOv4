namespace tensorflow {
    using System.Collections.Generic;
    using System.Globalization;
    using System.Linq;

    using tensorflow.keras;
    using tensorflow.keras.layers;

    static class Utils {
        public static void SetTrainableRecursive(ILayer parent, bool trainable) {
            parent.trainable = trainable;
            if (parent is IModel model) {
                foreach (ILayer nested in model.layers)
                    SetTrainableRecursive(nested, trainable);
            }
        }

        public static void FreezeAll(IModel model) => SetTrainableRecursive(model, false);
        public static void UnfreezeAll(IModel model) => SetTrainableRecursive(model, true);
    }
}
