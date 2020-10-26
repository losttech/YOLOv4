namespace tensorflow.keras.callbacks {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LostTech.Gradient;

    class TrainingLogger: Callback, ICallback {
        public override void on_epoch_end(int epoch, IDictionary<string, object?> logs) {
            string metrics = string.Join(";  ", logs.Select(entry =>
                $"{entry.Key}={((Tensor)entry.Value).numpy()}"));
            Console.WriteLine($"epoch {epoch} @ {DateTime.Now.ToLongTimeString()}:  {metrics}");
        }

        dynamic? ICallback.on_epoch_end(dynamic epoch, dynamic logs) {
            this.on_epoch_end((int)epoch, (IDictionary<string, object?>)logs);
            return null;
        }
    }
}
