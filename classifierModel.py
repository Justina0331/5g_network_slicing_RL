import pickle

class classifierModel:

    def get_infos_from_csv(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)

        filtered_col = 'flowStart flowEnd flowDuration min_piat max_piat \
        avg_piat std_dev_piat web_service f_avg_piat f_std_dev_piat b_flowStart \
        b_flowEnd b_flowDuration b_min_piat b_max_piat b_avg_piat b_std_dev_piat \
        flowEndReason f_flowStart f_flowEnd f_flowDuration f_min_piat f_max_piat'.split(' ')

        feats = [x for x in df.columns if x not in filtered_col]
        x = df[feats]
        #y = df['web_service']
        return x
    
    def get_model(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    
print("classifierModel_done")