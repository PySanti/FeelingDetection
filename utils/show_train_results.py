import matplotlib.pyplot as plt
def show_train_results(history):
    history_dict = history.history
    metrics = list(history_dict.keys())
    
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    loss_keys = [k for k in metrics if 'loss' in k]
    for key in loss_keys:
        color = 'b-' if 'val' not in key else 'r-'
        plt.plot(history_dict[key], color, label=key)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    acc_keys = [k for k in metrics if 'accuracy' in k or 'acc' in k]
    for key in acc_keys:
        color = 'b-' if 'val' not in key else 'r-'
        plt.plot(history_dict[key], color, label=key)
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
