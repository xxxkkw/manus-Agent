import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MVF.model.MVF_SleepNet import build_MVFSleepNet
from MVF.model.Utils import *

def detect():
    Path, _, cfgTrain, cfgModel = ReadConfig("MVF/config/config.config")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    fold = int(cfgTrain["fold"])
    context = int(cfgTrain["context"])
    optimizer = cfgTrain["optimizer"]
    learn_rate = float(cfgTrain["learn_rate"])
    GLalpha = float(cfgModel["GLalpha"])
    num_of_chev_filters = int(cfgModel["cheb_filters"])
    num_of_time_filters = int(cfgModel["time_filters"])
    time_conv_strides = int(cfgModel["time_conv_strides"])
    time_conv_kernel = int(cfgModel["time_conv_kernel"])
    num_block = int(cfgModel["num_block"])
    cheb_k = int(cfgModel["cheb_k"])
    dropout = float(cfgModel["dropout"])

    output_messages = ["睡眠阶段分析报告", "======================"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "sleep_output/")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(fold):

        Features = np.load(Path['Save'] + 'Feature_' + str(i) + '.npz', allow_pickle=True)
        val_feature = Features['val_feature']
        val_targets = Features['val_targets']
        
        val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)
        
        Features = np.load(Path['save'] + 'STFT_Feature_' + str(i) + '.npz', allow_pickle=True)
        val_STFT_adjacent = Features['val_feature']
        val_STFT_adjacent, _ = AddContext_SingleSub_STFT_GenFeature(val_STFT_adjacent, val_targets, 5)

        model = build_MVFSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                                time_conv_kernel, val_feature.shape[1:], val_STFT_adjacent.shape[1:], 
                                num_block, dropout, Instantiation_optim(optimizer, learn_rate), GLalpha)
        model.load_weights(Path['save'] + 'MVF_SleepNet_Best_' + str(i) + '.h5')
        predicts = model.predict([val_feature, val_STFT_adjacent])

        df = pd.DataFrame({
            'time_index': range(len(predicts)),
            'stage': np.argmax(predicts, axis=1)
        })
        
        stage_counts = df['stage'].value_counts(normalize=True).sort_index()
        stage_percent = stage_counts.reindex(range(5), fill_value=0)

        output_messages.extend([
            f"受试者 S{i+1} 睡眠阶段比例:",
            *[f"• {['W','N1','N2','N3','REM'][j]}: {stage_percent[j]:.2%}" for j in range(5)],
            ""
        ])

        subject_dir = os.path.join(output_dir, f"S{i+1}")
        os.makedirs(subject_dir, exist_ok=True)
        df.to_csv(os.path.join(subject_dir, f"S{i+1}_stages.csv"), index=False)

        
        plt.figure(figsize=(16, 9))  
        scatter = plt.scatter(df['time_index'], df['stage'], 
                             c=df['stage'], 
                             cmap=plt.cm.colors.ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']),
                             s=20, marker='o')
        
        plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.ylim(-0.5, 4.5)  
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.title(f'Subject S{i+1} - Sleep Stage Predictions')
        plt.xlabel('Time Points')
        plt.yticks([])
        
        handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=c, 
                            markersize=10) for c in ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']]
        plt.legend(handles, ['W','N1','N2','N3','REM'], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.text(0.01, 0.75, '\n'.join([f"{s}: {stage_percent[j]:.1%}" 
                                    for j,s in enumerate(['W','N1','N2','N3','REM'])]),
               transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, f"S{i+1}_predictions.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(16, 6))
        plt.plot(df['time_index'], df['stage'], 
                color='#1f77b4', linewidth=1.5, marker='o', markersize=2)
        
        plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.ylim(-0.5, 4.5)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.title(f'Subject S{i+1} - Sleep Stage Trends')
        plt.xlabel('Time Points')
        plt.ylabel('Sleep Stage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_dir, f"S{i+1}_trends.png"), bbox_inches='tight')
        plt.close()

    output_messages.append(f"\n分析结果以及图像保存路径: {output_dir}")
    return {
        "report": output_messages,
        "save_path": output_dir
    }

if __name__ == "__main__":
    result = detect()
    print("\n".join(result["report"]))