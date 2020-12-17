namespace tensorflow.keras.callbacks {
    using System;
    using System.Collections.Generic;
    internal class LearningRateLogger: Callback {
        public override void on_epoch_end(int epoch, IDictionary<string, object> logs) {
            Tensor learningRate = this.model.optimizer._decayed_lr(tf.float32);
            logs["lr"] = tf.keras.backend.eval(learningRate);
        }
    }
}
