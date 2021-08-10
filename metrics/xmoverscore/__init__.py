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
        BertRemap.__init__(self, model_name, None, mapping, device, do_lower_case, remap_size, embed_batch_size, alignment)

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
        train_size = 200000,
        align_batch_size = 5000,
        mine_batch_size = 5000000, 
        src_lang = "de",
        tgt_lang = "en",
        model_name="bert-base-multilingual-cased",
        monolingual_model_name=None,
        mt_model_name="facebook/mbart-large-cc25",
        mapping="UMD",
        do_lower_case=False,
        remap_size = 2000,
        embed_batch_size = 128,
        translate_batch_size = 16,
        nmt_weights = [0.8, 0.2],
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverNMTAlign.__init__(self, device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang,
                tgt_lang, mt_model_name, translate_batch_size, nmt_weights, use_cosine, mine_batch_size)
        BertRemap.__init__(self, model_name, monolingual_model_name, mapping, device, do_lower_case, remap_size,
                embed_batch_size, alignment)

class XMoverNMTLMBertAlignScore(XMoverNMTLMAlign, BertRemap):
    def __init__(
        self,
        device="cuda" if cuda_is_available() else "cpu",
        use_cosine = False,
        use_lm = False,
        alignment = "awesome",
        k = 20,
        n_gram = 1,
        knn_batch_size = 1000000,
        train_size = 200000,
        align_batch_size = 5000,
        mine_batch_size = 5000000,
        lm_weights = [1, 0.1],
        nmt_weights = [0.8, 0.2],
        src_lang = "de",
        tgt_lang = "en",
        model_name="bert-base-multilingual-cased",
        monolingual_model_name=None,
        mt_model_name="facebook/mbart-large-cc25",
        lm_model_name="gpt2",
        mapping="UMD",
        do_lower_case=False,
        remap_size = 2000,
        embed_batch_size = 128,
        translate_batch_size = 16,
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverNMTLMAlign.__init__(self, device, k, n_gram, knn_batch_size, train_size, align_batch_size, src_lang, tgt_lang,
                mt_model_name, translate_batch_size, nmt_weights, use_cosine, mine_batch_size, use_lm, lm_weights, lm_model_name)
        BertRemap.__init__(self, model_name, monolingual_model_name, mapping, device, do_lower_case, remap_size,
                embed_batch_size, alignment)

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
        lm_model_name="gpt2",
        mapping="UMD",
        device="cuda" if cuda_is_available() else "cpu",
        do_lower_case=False,
        use_cosine = False,
        use_lm = False,
        k = 20,
        n_gram = 1,
        embed_batch_size = 128,
        knn_batch_size = 1000000,
        align_batch_size = 5000,
        lm_weights = [1, 0.1]
    ):
        logging.info("Using device \"%s\" for computations.", device)
        XMoverLMAlign.__init__(self, device, k, n_gram, knn_batch_size, use_cosine, align_batch_size, use_lm,
                lm_weights, lm_model_name)
        BertRemapPretrained.__init__(self, model_name, None, mapping, device, do_lower_case, embed_batch_size)
