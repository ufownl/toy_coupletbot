import mxnet as mx

class Seq2seqLSTM(mx.gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout=0.5, **kwargs):
        super(Seq2seqLSTM, self).__init__(**kwargs)
        with self.name_scope():
            self._embed = mx.gluon.nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))
            self._attention = mx.gluon.nn.Dense(num_hidden, use_bias=False)
            self._encode = mx.gluon.rnn.LSTM(num_hidden, num_layers)
            self._decode = mx.gluon.rnn.LSTM(num_hidden, num_layers)
            self._dropout = mx.gluon.nn.Dropout(dropout)
            self._output = mx.gluon.nn.Dense(vocab_size)
        self._num_hidden = num_hidden

    def forward(self, source, target, hidden):
        output, hidden = self.encode(source, hidden)
        return self.decode(target, hidden, output)

    def encode(self, inputs, hidden):
        embed = self._embed(inputs)
        return self._encode(embed, hidden)

    def decode(self, inputs, hidden, enc_out):
        embed = self._embed(inputs)
        w_e = self._attention(enc_out.reshape((-1, self._num_hidden)))
        w_e = w_e.expand_dims(2)
        outputs = []
        for t in embed:
            output, hidden = self._decode_step(t, hidden, enc_out, w_e)
            outputs += [output]
        return mx.nd.concat(*outputs, dim=0), hidden

    def begin_state(self, *args, **kwargs):
        return self._encode.begin_state(*args, **kwargs)

    def _do_attention(self, hidden, enc_out, w_e):
        h = mx.nd.broadcast_axis(hidden[0][-1].expand_dims(0), axis=0, size=enc_out.shape[0])
        a = mx.nd.batch_dot(h.reshape((-1, self._num_hidden)).expand_dims(2), w_e, transpose_a=True)
        a = mx.nd.softmax(a.reshape((enc_out.shape[0], -1, 1)), axis=0)
        #print(a.reshape((-1,)))
        c = mx.nd.batch_dot(enc_out.swapaxes(0, 1), a.swapaxes(0, 1), transpose_a=True)
        return c.reshape((-1, self._num_hidden))

    def _decode_step(self, embed, hidden, enc_out, w_e):
        context = self._do_attention(hidden, enc_out, w_e)
        output, hidden = self._decode(mx.nd.concat(embed, context, dim=1).expand_dims(0), hidden)
        output = self._dropout(output)
        output = self._output(output.reshape((-1, self._num_hidden)))
        return output, hidden
