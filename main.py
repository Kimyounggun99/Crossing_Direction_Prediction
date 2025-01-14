import torch
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from build_dataset.dataprocessor import create_dataloaders, build_data, parse_args
from build_model.main_model import Transformer_based_model, GCN_based_model, Transformer_GCN_mixing_model

def evaluate_model(model, test_loader, device, args):
    """Evaluate the model on the test set and calculate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            center= torch.concat((batch['center'],batch['polar_angles']), axis=-1)
            key_3d= torch.concat((batch['key_points_positions'],batch['key_points_scores']), axis=-1)
            
            if args.model=='Transformer_based_model':
                B, T, V, C= key_3d.shape
                key_3d= key_3d.view(B,T,V*C)

            elif args.model== 'GCN_based_model':
                center= center.unsqueeze(-2)
                center= center.permute(0, 3, 1, 2)
                key_3d= key_3d.permute(0, 3, 1, 2)
            
            elif args.model== 'Transformer_GCN_mixing_model':
                key_3d= key_3d.permute(0, 3, 1, 2)
            
            pred=model(kp=key_3d.to(device), center= center.to(device))    


            # 예측값과 레이블 변환
            pred = pred.squeeze(-1).float()  # (batch_size, 1) → (batch_size,)
            pred_binary = (pred >= 0.5).float()
            labels = batch['C_labels'].to(device).float()

            # 모든 텐서가 1차원인지 확인 후 추가
            if pred_binary.ndim == 0:  # 스칼라 텐서인 경우
                pred_binary = pred_binary.unsqueeze(0)
            if labels.ndim == 0:  # 스칼라 텐서인 경우
                labels = labels.unsqueeze(0)

            all_preds.append(pred_binary.cpu())
            all_labels.append(labels.cpu())
    

    all_preds = torch.cat(all_preds)  # (num_samples,)
    all_labels = torch.cat(all_labels)  # (num_samples,)

    # NumPy 배열로 변환 후 sklearn 메트릭 계산
    all_preds = all_preds.numpy()
    all_labels = all_labels.numpy()
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, recall, precision, conf_matrix

def main():

    args= parse_args()

    train_data, test_data= build_data(args)

    train_loader, test_loader = create_dataloaders(train_data, test_data, args.bs, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dummy model for example

    if args.model=='Transformer_based_model':
        model= Transformer_based_model(device=device)
    elif args.model=='GCN_based_model':
        model= GCN_based_model(device=device)
    elif args.model=='Transformer_GCN_mixing_model':
        model= Transformer_GCN_mixing_model(device=device)
    else:
        breakpoint()

    # Set up logging
    
    if args.raw_point:
        log_file = f'{args.output_dir}/ablation/{args.model}_rawpoints_observ_{args.observation_time}sec.txt'
    else:
        log_file = f'{args.output_dir}/{args.experiment_type}/{args.model}_observ_{args.observation_time}sec.txt'
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Training started...")

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    patients=0
    if args.mode=='train':
        print("Training... ")
        # Training loop
        best_acc = 0
      
        
        if args.raw_point:
            best_model_path = f'{args.save_dir}/ablation/{args.model}_rawpoints_observ_{args.observation_time}sec.pth'
        else:
            best_model_path = f'{args.save_dir}/{args.experiment_type}/{args.model}_observ_{args.observation_time}sec.pth'
        # Training loop
        for epoch in range(args.epoch):  # Number of epochs
            model.train()
            
            for batch in train_loader:

                center= torch.concat((batch['center'],batch['polar_angles']), axis=-1)
                key_3d= torch.concat((batch['key_points_positions'],batch['key_points_scores']), axis=-1)
        
                if args.model=='Transformer_based_model':
                    B, T, V, C= key_3d.shape
                    key_3d= key_3d.view(B,T,V*C)
                elif args.model== 'GCN_based_model':
                    center= center.unsqueeze(-2)
                    center= center.permute(0, 3, 1, 2)
                    key_3d= key_3d.permute(0, 3, 1, 2)
                elif args.model== 'Transformer_GCN_mixing_model':
                    key_3d= key_3d.permute(0, 3, 1, 2)
                else:
                    breakpoint()

                pred=model(kp=key_3d.to(device), center= center.to(device))

                labels = batch['C_labels'].to(device)  # (batch_size,)
          
                labels = labels.unsqueeze(-1).float()
                if len(pred)==1:
                    pred= pred.unsqueeze(-1)
                pred = pred.float()
                loss = loss_fn(pred, labels)  # Assuming labels are class indices

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch::{epoch}")
                # Evaluate on the test set every N epochs
            
            accuracy, f1, recall, precision, conf_matrix = evaluate_model(model, test_loader, device,args)
            log_message = (
                f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                f"Test Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.4f}, "
                f"Recall: {recall:.4f}, Precision: {precision:.4f}\n"
                f"Confusion Matrix:\n{conf_matrix}"
            )
            logging.info(log_message)
            print(log_message)
            if accuracy<=best_acc:
                patients+=1
            # Save the best model
            else:
                patients=0
                best_acc = accuracy
                
                log_message = f"Best model saved with Acc: {best_acc}"
                logging.info(log_message)
                print(log_message)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, best_model_path)
            if patients>20:
                print(f"Early stop, Epoch:{epoch}")
                break
        print(f"Training completed.")

    else:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model weights were successfuly uploaded!!!')
        print('Testing...\n')
        accuracy, f1, recall, precision, conf_matrix= evaluate_model(model, test_loader, device, args)
        print(f"Model:: {args.model}, Acc: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}, Confusion_matrix: {conf_matrix}")



if __name__=='__main__':
    
    main()