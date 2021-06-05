from .embed import *
from .align import *
from torch.cuda import is_available as cuda_is_available

class XMoverBertAlignScore(XMoverAlign, BertRemap):
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        use_cosine = False,
        alignment = "awesome",
        k = 20,
        n_gram = 1,
        remap_size = 2000,
        embed_batch_size = 128,
        knn_batch_size = 1000000,
        align_batch_size = 5000
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverAlign.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        BertRemap.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size, alignment)

class XMoverVecMapAlignScore(XMoverAlign, VecMapEmbed):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_cosine = False,
        k = 20,
        n_gram = 1,
        knn_batch_size = 1000000,
        src_lang = "de",
        tgt_lang = "en",
        batch_size = 5000,
        align_batch_size = 5000
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverAlign.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size)
        VecMapEmbed.__init__(self, device, src_lang, tgt_lang, batch_size)

class XMoverNMTBertAlignScore(XMoverNMTAlign, BertRemap):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_cosine = False,
        alignment = "awesome",
        k = 20,
        n_gram = 1,
        knn_batch_size = 1000000,
        train_size = 500000,
        align_batch_size = 5000,
        src_lang = "de",
        tgt_lang = "en",
        model_name="bert-base-multilingual-cased",
        mt_model_name="facebook/mbart-large-cc25",
        mapping="UMD",
        do_lower_case=False,
        remap_size = 2000,
        embed_batch_size = 128,
        translate_batch_size = 16,
        ratio = 0.5
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverNMTAlign.__init__(self, device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang,
                tgt_lang, mt_model_name, translate_batch_size, ratio, use_cosine)
        BertRemap.__init__(self, model_name, mapping, device, do_lower_case, remap_size, embed_batch_size, alignment)

class XMoverScore(XMoverLMAlign, BertRemapPretrained):
    """
    The original XMoverScore implementation. Be careful, remapping matrices
    were trained on parallel data! Provided out of convienence to compare the
    preformance of self-learning remapping approaches to the supervised
    original.
    """
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        use_cosine = False,
        use_lm = False,
        k = 20,
        n_gram = 1,
        model_batch_size = 128,
        knn_batch_size = 1000000,
        align_batch_size = 5000,
        lm_weights = [1, 0.1]
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverLMAlign.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size, model_batch_size,
            use_lm, lm_weights)
        BertRemapPretrained.__init__(self, model_name, mapping, device, do_lower_case, model_batch_size)
