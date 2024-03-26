import pickle
import pandas as pd

'''
拿學長的分類器來用，應該不用改
'''
class classifierModel:

    # 將 csv 中的 slice 欄位轉換為整數
    def slice_str_to_int(string):
        slices_mapping = { 'slice_A': 0, 'slice_B': 1, 'slice_C': 2, 'slice_1': 0, 'slice_2': 1, 'slice_3': 2 }
        return slices_mapping[string]

    def get_infos_from_csv(csv_path):

        df = pd.read_csv(csv_path)
        filtered_col = 'flowStart flowEnd flowDuration min_piat max_piat \
        avg_piat std_dev_piat web_service f_avg_piat f_std_dev_piat b_flowStart \
        b_flowEnd b_flowDuration b_min_piat b_max_piat b_avg_piat b_std_dev_piat \
        flowEndReason f_flowStart f_flowEnd f_flowDuration f_min_piat f_max_piat'.split(' ')

        feats = [x for x in df.columns if x not in filtered_col]
        x = df[feats]
        y = df['web_service']  #real_slice
        return x, y
    
    def get_model(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    