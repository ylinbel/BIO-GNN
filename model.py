import torch
import helper
import random
import uuid
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import time
from torch.nn import Sequential, Linear, ReLU
import torch.nn
import matplotlib.pyplot as plt
from echoes import ESNRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#These two options shoul be seed to ensure reproducible (If you are using cudnn backend)
#https://pytorch.org/docs/stable/notes/randomness.html
#We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
np.random.seed(35813)
torch.manual_seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_WEIGHT_BACKUP_PATH = "./output"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"



def show_image(img, i, score):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("fold " + str(i) + " Frobenious distance: " +  "{:.2f}".format(score))
    plt.axis('off')
    plt.show()
    
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


class DGN(torch.nn.Module):
    def __init__(self, MODEL_PARAMS):
        super(DGN, self).__init__()
        self.model_params = MODEL_PARAMS
        
        nn = Sequential(Linear(self.model_params["Linear1"]["in"], self.model_params["Linear1"]["out"]), ReLU())
        self.conv1 = NNConv(self.model_params["conv1"]["in"], self.model_params["conv1"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(self.model_params["Linear2"]["in"], self.model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(self.model_params["conv2"]["in"], self.model_params["conv2"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(self.model_params["Linear3"]["in"], self.model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(self.model_params["conv3"]["in"], self.model_params["conv3"]["out"], nn, aggr='mean')
        
        self.W_out = torch.nn.Parameter(torch.randn(self.model_params["N_ROIs"], self.model_params["N_ROIs"] + 1), requires_grad=True)
        
    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format) 
                        

        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        
        repeated_out = x.repeat(self.model_params["N_ROIs"],1,1)
        repeated_t   =  torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)
        
        return cbt
    
    @staticmethod
    def generate_subject_biased_cbts(model, train_data):
        """
            Generates all possible CBTs for a given training set.
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = np.zeros((model.model_params["N_ROIs"],model.model_params["N_ROIs"], len(train_data)))
        train_data = [d.to(device) for d in train_data]
        for i, data in enumerate(train_data):
            cbt = model(data)
            cbts[:,:,i] = np.array(cbt.cpu().detach())
        
        return cbts
        
    @staticmethod
    def generate_cbt_median(model, train_data):
        """
            Generate optimized CBT for the training set (use post training refinement)
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = []
        train_data = [d.to(device) for d in train_data]
        for data in train_data:
            cbt = model(data)
            cbts.append(np.array(cbt.cpu().detach()))
        final_cbt = torch.tensor(np.median(cbts, axis = 0), dtype = torch.float32).to(device)
        
        return final_cbt   
    
    @staticmethod
    def w_out_error(generated_cbt, x_train, y_train):
        normalized_cbt = helper.normalize_matrix(generated_cbt.numpy())
        esn = ESNRegressor(
            spectral_radius=0.99,
            input_scaling=1e-6,
            leak_rate=1,
            bias=0,
            W=normalized_cbt,
            store_states_train=True,
        )
        esn.fit(X=x_train, y=y_train)
        
        y_hat = esn.activation_out(torch.tensor(esn.states_train_, dtype=torch.float32) @ generated_cbt.detach().numpy())
        w_out_loss = torch.mean((torch.tensor(y_hat, dtype=torch.float32) - torch.tensor(y_train, dtype=torch.float32)) ** 2)
        
        return w_out_loss
     
    @staticmethod
    def mean_frobenious_distance(generated_cbt, test_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views)
            Args:
                generated_cbt: trained DGN model
                test_data: list of data objects
        """
        frobenius_all = []
        for data in test_data:
            views = data.con_mat
            for index in range(views.shape[2]):
                diff = torch.abs(views[:,:,index] - generated_cbt)
                diff = diff*diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)
    
    @staticmethod
    def biological_fingerprint(generated_cbt, x_train, y_train, x_test, y_test):
        normalized_cbt = helper.normalize_matrix(generated_cbt.numpy())
        esn = ESNRegressor(
            spectral_radius=0.99,
            input_scaling=1e-6,
            leak_rate=1,
            bias=0,
            W=normalized_cbt,
            # store_states_train=True,
            # regression_method="ridge",
        )
        esn.fit(X=x_train, y=y_train)
        y_pred = esn.predict(x_test)
        fingerprint = helper.memory_capacity_cul_sum(y_test, y_pred)
        return fingerprint
    
    @staticmethod
    def train_model(X, model_params, n_max_epochs, early_stop, model_name, random_sample_size = 10, n_folds = 5):
        """
            Trains a model for each cross validation fold and 
            saves all models along with CBTs to ./output/<model_name> 
            Args:
                X (np array): dataset (train+test) with shape [N_Subjects, N_ROIs, N_ROIs, N_Views]
                n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
                early_stop (bool): if set true, model will stop training when overfitting starts.
                model_name (string): name for saving the model
                random_sample_size (int): random subset size for SNL function
                n_folds (int): number of cross validation folds
            Return:
                models: trained models 
        """
        models = []

        save_path = MODEL_WEIGHT_BACKUP_PATH + "/" + model_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        model_id = str(uuid.uuid4())
        with open(save_path + "model_params.txt", 'w') as f:
            print(model_params, file=f)
        
        CBTs = []
        scores = []
        
        lambda1 = 300
        lambda2 = 10 
        for i in range(n_folds):
            torch.cuda.empty_cache() 
            print("********* FOLD {} *********".format(i))
            train_data, test_data, train_mean, train_std = helper.preprocess_data_array(X, number_of_folds=n_folds, current_fold_id=i)
            
            test_casted = [d.to(device) for d in helper.cast_data(test_data)]
            loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean))*len(train_data)), dtype = torch.float32)
            loss_weightes = loss_weightes.to(device)
            train_casted = [d.to(device) for d in helper.cast_data(train_data)]

            model = DGN(model_params)
            model = model.to(device)
            
            # cbt loss stop training
            cbt_loss_history = []
            cbt_loss_converged = False
            cbt_convergence_threshold = 0.01
            cbt_patience = 3
            bio_patience = 3
            min_reservoir_loss = float('inf')
            min_bio_loss = float('inf')
            
            cbt_loss_weight = 1.0
            reservoir_loss_weight = 100.0
            prev_cbt_loss_avg = float('inf')
            loss_weight_adjustment_factor = 5 
            
            # W_out loss setup
            reservoir_model = LinearRegressionModel(36, 1).to(device)
            
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(reservoir_model.parameters()), lr=model_params["learning_rate"], weight_decay= 0.00)

            targets = [torch.tensor(tensor, dtype = torch.float32).to(device) for tensor in train_data]
            # `test_errors` is a list that keeps track of the test representation error (mean
            # Frobenius distance) at each epoch during training. It is used for early stopping and to
            # monitor the model's performance over time.
            test_errors = []
            tick = time.time()
            
            x_train, y_train, x_test, y_test = helper.generate_data()
            
            for epoch in range(n_max_epochs):
                model.train()
                biological_fingerprints = []
                w_out_losses = []
                cbt_losses = [] 
                bio_losses = []
                total_loss = 0
                for data in test_casted:
                    # loss of cbt
                    #Compose Dissimilarity matrix from network outputs
                    cbt = model(data)
                    views_sampled = random.sample(targets, random_sample_size)
                    sampled_targets = torch.cat(views_sampled, axis = 2).permute((2,1,0))
                    expanded_cbt = cbt.expand((sampled_targets.shape[0],model_params["N_ROIs"],model_params["N_ROIs"]))
                    diff = torch.abs(expanded_cbt - sampled_targets) #Absolute difference
                    sum_of_all = torch.mul(diff, diff).sum(axis = (1,2)) #Sum of squares
                    l = torch.sqrt(sum_of_all)  #Square root of the sum
                    cbt_loss = (l * loss_weightes[:random_sample_size * model_params["n_attr"]]).mean()
                    cbt_losses.append(cbt_loss.item())
                    
                    
                    # loss of reservoir
                    cbt_normalized = helper.normalize_matrix(cbt.cpu().detach().numpy())
                    esn = ESNRegressor(
                        spectral_radius=0.99,
                        input_scaling=1e-6,
                        leak_rate=1,
                        bias=0,
                        W=cbt_normalized,
                        store_states_train=True,
                    )
                    esn.fit(X=x_train, y=y_train)
                    
                    reservoir_output = reservoir_model(torch.tensor(esn.full_states_, dtype=torch.float32))
                    w_out_loss = torch.mean((reservoir_output - torch.tensor(y_train, dtype=torch.float32, device=device)) ** 2)
                    w_out_losses.append(w_out_loss.item())
                    
                    # biological loss
                    y_pred = esn.predict(x_test)
                    fingerprint = helper.memory_capacity_cul_sum(y_test, y_pred)
                    biological_fingerprints.append(fingerprint)
                    
                    if not cbt_loss_converged:
                        # total_loss += (cbt_loss + lambda1 * w_out_loss)
                        total_loss = cbt_loss_weight * cbt_loss + reservoir_loss_weight * w_out_loss
                    total_loss += w_out_loss
                    
                    cbt_losses.append(cbt_loss.item())
                    
            	#Backprob                
                avg_cbt_loss = sum(cbt_losses) / len(cbt_losses)
                cbt_loss_history.append(avg_cbt_loss)
                if len(cbt_loss_history) > cbt_patience:
                    recent_cbt_loss_improvement = cbt_loss_history[-cbt_patience] - cbt_loss_history[-1]
                    if recent_cbt_loss_improvement < cbt_convergence_threshold:
                        cbt_loss_converged = True
                        
                if avg_cbt_loss >= prev_cbt_loss_avg:
                    reservoir_loss_weight *= loss_weight_adjustment_factor  # Increase reservoir loss weight
                else:
                    reservoir_loss_weight /= loss_weight_adjustment_factor
                    
                
                optimizer.zero_grad()                
                total_loss.backward()                
                optimizer.step()
                
                #Track the loss
                if epoch % 10 == 0:
                    if cbt_loss_converged:
                        print("CBT loss has converged.")
                    cbt = DGN.generate_cbt_median(model, test_casted)
                    rep_loss = DGN.mean_frobenious_distance(cbt, train_casted)
                    w_out_loss = DGN.w_out_error(cbt, x_train, y_train)
                    w_out_loss = round(np.mean(w_out_losses), 4)
                    w_out_losses.clear()
                    cbt_fingerprint = DGN.biological_fingerprint(cbt, x_train, y_train, x_test, y_test) 
                    
                    bio_loss = ((biological_fingerprints - cbt_fingerprint) ** 2).mean()
                    bio_losses.append(bio_loss.item())
                    biological_fingerprints.clear()
                    tock = time.time()
                    time_elapsed = tock - tick
                    tick = tock
                    rep_loss = float(rep_loss)
                    
                    current_error = rep_loss + w_out_loss * lambda1 + bio_loss * lambda2
                    test_errors.append(current_error)
                    print("Epoch: {}  |  cbt loss : {:.2f} | reservoir loss : {:.4f} | bio loss : {:.5f} | total loss: {:.2f} | median cbt mc {:.2f} | Time Elapsed: {:.2f} | ".format(epoch, rep_loss, w_out_loss, bio_loss * lambda2, current_error, cbt_fingerprint, time_elapsed))

                    if cbt_loss_converged:
                        # Early stopping controls
                        bio_improved = False
                        if bio_loss < min_bio_loss:
                            min_bio_loss = bio_loss
                            bio_improved = True
                        
                        # Check for improvement in reservoir loss
                        reservoir_improved = False
                        if w_out_loss < min_reservoir_loss:
                            min_reservoir_loss = w_out_loss
                            reservoir_improved = True
                        
                        # Save model if there's improvement in either loss
                        if reservoir_improved or bio_improved:
                            no_improvement_count = 0
                            torch.save(model.state_dict(), TEMP_FOLDER + f"/best_model_epoch_{epoch}.model")
                        else:
                            no_improvement_count += 1

                        # Early stopping based on lack of improvement in both losses
                        if no_improvement_count >= bio_patience:
                            print("Early Stopping triggered based on lack of improvement in reservoir and biological losses.")
                            break
                
                
        	#Restore best model so far
            try:
                restore = "./temp/weight_" + model_id + "_" + str(min(test_errors))[:5] + ".model"
                model.load_state_dict(torch.load(restore))
            except:
                pass
            torch.save(model.state_dict(), save_path + "fold" + str(i) + ".model")
            models.append(model)
            #Generate and save refined CBT
            cbt = DGN.generate_cbt_median(model, test_casted)
            well_trained_memory_fingerprint = DGN.biological_fingerprint(cbt, x_train, y_train, x_test, y_test) 
            rep_loss = DGN.mean_frobenious_distance(cbt, test_casted)
            w_out_loss = DGN.w_out_error(cbt, x_train, y_train)
            cbt = cbt.cpu().numpy()
            CBTs.append(cbt)
            np.save( save_path + "fold" + str(i) + "_cbt", cbt)
            #Save all subject biased CBTs
            all_cbts = DGN.generate_subject_biased_cbts(model, test_casted)
            np.save(save_path + "fold" + str(i) + "_all_cbts", all_cbts)
            scores.append(float(rep_loss))
            print("FINAL RESULTS  REP: {}, W_OUT: {}, Memory Fingerprint: {}".format(rep_loss, w_out_loss, well_trained_memory_fingerprint))
            #Clean interim model weights
            helper.clear_dir(TEMP_FOLDER)
                        
        for i, cbt in enumerate(CBTs):
            show_image(cbt, i, scores[i])
        return models