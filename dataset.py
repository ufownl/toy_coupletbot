import mxnet as mx
from vocab import Vocabulary

def load_conversations(path):
    with open(path, "r") as f:
        raw = f.read()

    dataset = []
    for conv in raw.split("E\n")[1:]:
        qa = conv.split("\n")
        if len(qa) >= 2:
            dataset.append((qa[0][2:], qa[1][2:]))

    return dataset


def dataset_filter(dataset, sequence_length):
    return [(src, tgt) for src, tgt in dataset if len(src) <= sequence_length and len(tgt) <= sequence_length]


def make_vocab(dataset):
    chars = sorted(list(set([ch for conv in dataset for sent in conv for ch in sent])))
    return Vocabulary(chars)


def tokenize(dataset, vocab):
    return [tuple([vocab.char2idx(ch) for ch in sent] for sent in conv) for conv in dataset]


def rnn_buckets(dataset, buckets):
    min_len = -1
    for max_len in buckets:
        bucket = [(src, tgt) for src, tgt in dataset if len(src) > min_len and len(src) <= max_len]
        min_len = max_len
        if len(bucket) > 0:
            yield bucket, max_len


def rnn_batches(dataset, vocab, batch_size, sequence_length, ctx):
    src_tok, tgt_tok = zip(*dataset)
    src_tok, tgt_tok = list(src_tok), list(tgt_tok)
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    for i in range(0, batches):
        start = i * batch_size
        src_bat = mx.nd.array(_pad_batch(src_tok[start: start + batch_size], vocab, sequence_length), ctx=ctx)
        src_bat = mx.nd.reverse(src_bat, axis=1)
        tgt_bat = mx.nd.array(_pad_batch(_add_sent_prefix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
        lbl_bat = mx.nd.array(_pad_batch(_add_sent_suffix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
        yield src_bat.T, tgt_bat.T, lbl_bat.T.reshape((-1,))


def pad_sentence(sent, vocab, buckets):
    min_len = -1
    for max_len in buckets:
        if len(sent) > min_len and len(sent) <= max_len:
            return sent + [vocab.char2idx("<PAD>")] * (max_len - len(sent))
        min_len = max_len
    return sent


def _add_sent_prefix(batch, vocab):
    return [[vocab.char2idx("<GO>")] + sent for sent in batch]


def _add_sent_suffix(batch, vocab):
    return [sent + [vocab.char2idx("<EOS>")] for sent in batch]


def _pad_batch(batch, vocab, seq_len):
    return [sent + [vocab.char2idx("<PAD>")] * (seq_len - len(sent)) for sent in batch]


if __name__ == "__main__":
    dataset = load_conversations("data/couplets.conv")
    print("dataset size: ", len(dataset))
    dataset = dataset_filter(dataset, 32)
    print("filtered dataset size: ", len(dataset))
    print("dataset preview: ", dataset[:10])
    vocab = make_vocab(dataset)
    print("vocab size: ", vocab.size())
    dataset = tokenize(dataset, vocab)
    print("tokenize dataset preview: ", dataset[:10])
    print("buckets preview: ", [(len(bucket), seq_len) for bucket, seq_len in rnn_buckets(dataset, [2, 4, 8, 16, 32])])
    print("batch preview: ", next(rnn_batches(dataset, vocab, 4, 32, mx.cpu())))
