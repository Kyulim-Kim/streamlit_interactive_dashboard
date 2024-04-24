from common.env import *


class PATH:
    root   = abspath(dirname(dirname(dirname(__file__))))
    script = join(root, 'KKL_diabetes_prediction_analysis', 'script')
    data   = join(root, 'KKL_diabetes_prediction_analysis', 'data')
    proc   = join(data, "processed")

    family        = join(data, "familyxx", "familyxx")
    sample_child  = join(data, "samchild", "samchild")
    sample_adult  = join(data, "samadult", "samadult")
    
    family_final_data = join(data, "familyxx", "familyxx_final.feather")
    sample_adult_final_data = join(data, "samadult", "samadult_final.feather")
    sample_child_final_data = join(data, "samchild", "samchild_final.feather")
    final = join(data, 'metas.pickle')

    @classmethod
    def get(cls, data_id):
        rst = {}
        if data_id not in ('family', 'sample_child', 'sample_adult'):
            raise ValueError(f"Invalid data_id: {data_id}")

        src = getattr(cls, f"{data_id}")
        rst['summary']    = f"{src}_summary.pdf"
        rst['layout']     = f"{src}_layout.pdf"
        rst['metadata']   = f"{src}_metadata.csv"
        rst['data']       = f"{src}.csv"
        return rst

    @classmethod
    def get_proc(cls):
        return {
            # before preprocessing
            'raw_data': join(cls.proc, "raw_data.csv"),
            'raw_metadata': join(cls.proc, "raw_metadata.csv"),

            # after preprocessing
            'train': join(cls.proc, "train.ftr"),
            'val': join(cls.proc, "val.ftr"),
            'test': join(cls.proc, "test.ftr"),
            'metadata': join(cls.proc, "metadata.csv")
        }


class PARAMS:
    figsize = (28, 4)
    seed    = 42
    target  = 'DIBEV1'
    target2 = 'CCONDRR6'
