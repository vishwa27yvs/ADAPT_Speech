# Commented out IPython magic to ensure Python compatibility.
# %%capture
# 
# !pip install git+https://github.com/huggingface/datasets.git
# !pip install git+https://github.com/huggingface/transformers.git
# !pip install jiwer
# !pip install torchaudio
# !pip install librosa
# 
# # Monitor the training process
# # !pip install wandb

# Commented out IPython magic to ensure Python compatibility.
# %env LC_ALL=C.UTF-8
# %env LANG=C.UTF-8
# %env TRANSFORMERS_CACHE=/content/cache
# %env HF_DATASETS_CACHE=/content/cache
# %env CUDA_LAUNCH_BLOCKING=1

# # Uncomment this part if you want to setup your wandb project

# %env WANDB_WATCH=all
# %env WANDB_LOG_MODEL=1
# %env WANDB_PROJECT=YOUR_PROJECT_NAME
# !wandb login YOUR_API_KEY --relogin

"""## Prepare Data for Training"""

# Place your training data under train.csv and testing data under test.csv
# The data should consist of 2 columns, path: containing the path to the audio file, emotion: emotion label in string or number format

# Loading the created dataset using datasets
from datasets import load_dataset, load_metric


data_files = {
    "train": "/content/data/train.csv", 
    "validation": "/content/data/test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print("Dataset Info")
print(train_dataset)
print(eval_dataset)

# We need to specify the input and output column
input_column = "path"
output_column = "emotion"

# we need to distinguish the unique labels in our SER dataset
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")

from transformers import AutoConfig, Wav2Vec2Processor

model_name_or_path = "facebook/wav2vec2-large-xlsr-53"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

from transformers import Wav2Vec2FeatureExtractor

# In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal (given by hugging face).
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

from transformers import BatchFeature
import random

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)


    # UNCOMMENT THE BELOW CROP SECTION FOR SHEMO
    # org_size = speech_array.shape[1]
    # crop_size = 2* sampling_rate
    # if org_size > crop_size:
    #     start = random.randint(0, org_size - crop_size)
    #     speech_array = speech_array[:,start : start + crop_size]

    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)    
    speech = resampler(speech_array).squeeze().numpy()

    #  UNCOMMENT THE FOLLOWING LINE FOR EMOVO
    #speech = resampler(speech_array[0]).squeeze().numpy()

    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):

    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result


train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=2,
    batched=True,
    num_proc=1
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=2,
    batched=True,
    num_proc=1
)

"""Great, now we've successfully read all the audio files, resampled the audio files to 16kHz, and mapped each audio to the corresponding label.

## Model

Before diving into the training part, we need to build our classification model based on the merge strategy.
"""

import random 
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

set_seed(101)

import manifolds

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# For information bottleneck
config.ib = True
config.ib_dim = 512
config.hidden_mlp_dim = (1024 + config.ib_dim)//2
#config.beta=1e-5
config.beta=1e-5
config.sample_size = 5
config.activation="relu"

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2ClassificationHeadVib(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.ib_dim, config.ib_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        #self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.out_proj = nn.Linear(config.ib_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassificationVib(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.activation = config.activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}

        self.wav2vec2 = Wav2Vec2Model(config)
        if True:
            #self.kl_annealing = config.kl_annealing
            self.ib = config.ib
            self.ib_dim = config.ib_dim
            self.hidden_dim = config.hidden_mlp_dim
            intermediate_dim = (self.hidden_dim+config.hidden_size)//2
            print("**intermediate dim**")
            print(intermediate_dim)
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = config.beta
            self.sample_size = config.sample_size
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))

            #self.classifier = nn.Linear(self.ib_dim, self.config.num_labels)
        self.classifier = Wav2Vec2ClassificationHeadVib(config)

        self.init_weights()

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        #if sampling_type == "iid":
        if self.training:
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits

    def sampled_loss(self, logits, mean_logits, labels, sampling_type):

        if self.training:
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        final_outputs = {}
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        if self.ib:
          # Loss computed differently for kl-divergence
          pooled_output=self.mlp(hidden_states)
          batch_size = pooled_output.shape[0]
          mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
          mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
          std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
          kl_loss = self.kl_div(mu, std, mu_p, std_p)
          z = self.reparameterize(mu, std)
          final_outputs["z"] = mu

          loss = None
          sampling_type="iid"
          sampled_logits, logits = self.get_logits(z, mu, sampling_type)

          if labels is not None:
              ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
              loss= ce_loss + (self.beta*kl_loss)

        else:
          logits = self.classifier(hidden_states)

          loss = None
          if labels is not None:
              if self.config.problem_type is None:
                  if self.num_labels == 1:
                      self.config.problem_type = "regression"
                  elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                      self.config.problem_type = "single_label_classification"
                  else:
                      self.config.problem_type = "multi_label_classification"

              if self.config.problem_type == "regression":
                  loss_fct = MSELoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels)
              elif self.config.problem_type == "single_label_classification":
                  loss_fct = CrossEntropyLoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
              elif self.config.problem_type == "multi_label_classification":
                  loss_fct = BCEWithLogitsLoss()
                  loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Hyperbolic function
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from layers.hyp_layers import HypLinear	
import manifolds

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

class Wav2Vec2ClassificationHeadVibHyperbolic(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.ib_dim, config.ib_dim)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.ib_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassificationVibHyperbolic(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.activation = config.activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}

        self.wav2vec2 = Wav2Vec2Model(config)
        if True:
            self.ib = config.ib
            self.ib_dim = config.ib_dim
            self.hidden_dim = config.hidden_mlp_dim
            intermediate_dim = (self.hidden_dim+config.hidden_size)//2
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = config.beta
            self.sample_size = config.sample_size

            hyperbolicity = 0.24
            a = (0.144/hyperbolicity)**2

            # For running HVIB 
            #self.c = 1.0

            # For running HVIB-C
            #self.c = (0.144/hyperbolicity)**2

            # For running ADAPT-VIB
            self.c = torch.nn.Parameter(torch.tensor(a,device="cuda"), requires_grad= True)

            self.manifold = getattr(manifolds, 'PoincareBall')()
            self.emb2mu = HypLinear(getattr(manifolds, 'PoincareBall')(),self.hidden_dim, self.ib_dim,self.c,0.2,False)	
            self.emb2std = HypLinear(getattr(manifolds, 'PoincareBall')(),self.hidden_dim, self.ib_dim,self.c,0.2,False)
            ##

            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))

        self.classifier = Wav2Vec2ClassificationHeadVibHyperbolic(config)

        self.init_weights()

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        #std = torch.nn.functional.softplus(emb2std(emb))
        std = emb2std(emb)	
        std = self.manifold.logmap0(std,self.c)	
        std = self.manifold.proj_tan0(std,self.c)		
        std = torch.nn.functional.softplus(std)
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        #if sampling_type == "iid":
        if self.training:
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits

    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        #if sampling_type == "iid":
        if self.training:
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        final_outputs = {}
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        if self.ib:
          # print("*here*")
          print(self.c)
          # Loss computed differently for kl-divergence
          pooled_output=self.mlp(hidden_states)
          batch_size = pooled_output.shape[0]
          
          # Hyperbolic 
          self.emb2mu.c= self.c
          self.emb2std.c= self.c

          pooled_output = self.manifold.proj_tan0(pooled_output,self.c)	
          pooled_output = self.manifold.expmap0(pooled_output,self.c)

          mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)

          mu = self.manifold.logmap0(mu,self.c)	
          mu = self.manifold.proj_tan0(mu,self.c)
          ##

          mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
          std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
          kl_loss = self.kl_div(mu, std, mu_p, std_p)
          z = self.reparameterize(mu, std)
          final_outputs["z"] = mu

          loss = None
          sampling_type="iid"
          sampled_logits, logits = self.get_logits(z, mu, sampling_type)


          if labels is not None:
              ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
              loss= ce_loss + (self.beta*kl_loss)

        else:
          logits = self.classifier(hidden_states)

          loss = None
          if labels is not None:
              if self.config.problem_type is None:
                  if self.num_labels == 1:
                      self.config.problem_type = "regression"
                  elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                      self.config.problem_type = "single_label_classification"
                  else:
                      self.config.problem_type = "multi_label_classification"

              if self.config.problem_type == "regression":
                  loss_fct = MSELoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels)
              elif self.config.problem_type == "single_label_classification":
                  loss_fct = CrossEntropyLoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
              elif self.config.problem_type == "multi_label_classification":
                  loss_fct = BCEWithLogitsLoss()
                  loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""## Training

The data is processed so that we are ready to start setting up the training pipeline. We will make use of huggingface's [Trainer]

- Define a data collator. In contrast to most NLP models, 
   XLSR-Wav2Vec2 has a much larger input length than output length. *E.g.*, a sample of input length 50000 
   has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training 
   batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and 
   not the overall longest sample. Therefore, fine-tuning XLSR-Wav2Vec2 requires a special padding data collator, which we will define below

- Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a `compute_metrics` function accordingly

- Load XLSR-53

- Define the training configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

is_regression = False

import numpy as np
from transformers import EvalPrediction

# one of the callbacks which are called
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# define call back for beta
from transformers import TrainerCallback
class BetaIncreaseCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, model, **kwargs):
        curr_epoch = state.epoch +1 
        model.beta = min(1,curr_epoch*(1e-5))
        print(model.beta)

from transformers import TrainerCallback
class PrintCurvature(TrainerCallback):
    def on_evaluate(self, args, state, control, model, **kwargs):
        curr_epoch = state.epoch +1 
        print(model.c)

from transformers import PreTrainedModel

def model_init():
   return Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path,config=config)

def model_init_vib():
   return Wav2Vec2ForSpeechClassificationVib.from_pretrained(model_name_or_path,config=config)

def model_init_hyperbolic_vib():
   return Wav2Vec2ForSpeechClassificationVibHyperbolic.from_pretrained(model_name_or_path,config=config)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    num_train_epochs=8.0,
    seed=101,
    save_steps=20,
    eval_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
)

from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer,
    is_apex_available,
)

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

trainer_base = Trainer(
    #model=model,
    model_init=model_init,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

trainer_vib = Trainer(
    model_init=model_init_vib,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # adding beta annealing callback
    callbacks=[BetaIncreaseCallback],
    tokenizer=processor.feature_extractor,
)

trainer_hyp_vib = Trainer(
    model_init=model_init_hyperbolic_vib,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # adding beta annealing callback, and curvature print
    callbacks=[BetaIncreaseCallback,PrintCurvature],
    tokenizer=processor.feature_extractor,
)

# Training
trainer_hyp_vib.train()
