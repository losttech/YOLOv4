namespace tensorflow.keras.losses {
    using System;
    class ZeroLoss : Loss {
        public override dynamic call(object y_true, object y_pred)
            => tf.constant_scalar(0f);

        public override dynamic call_dyn(object y_true, object y_pred)
            => throw new NotSupportedException();
    }
}
