import numpy as np

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def data_loader(seq_len):
    """
    Loads data into readable format
    Args:
        -seq_len: length of data sequence
    Returns:
        -loaded: loaded data set
    """
    data = np.loadtxt("data", delimiter=",")

    data = MinMaxScaler(data)

    loaded = []

    for i in range(len(data)  - seq_len):
       loaded.append(data[i:i+seq_len])
    
    np.random.shuffle(loaded)

    return loaded
    


def synthetic_data(num, seq_len):
    """
    Generates synthetic I/Q samples as 48k rate sampling
    """
    data = []
    I = np.random.uniform(-1, 1)
    Q =  np.random.uniform(-1, 1)
    for i in range(num):
        freq = np.random.uniform(0, 0.1)            
        phase = np.random.uniform(0, 0.1)
        temp = []
        Is = [I*np.sin(freq*j / 48000 + phase) for j in range(seq_len)]
        Qs = [Q*np.cos(freq*j / 48000 + phase) for j in range(seq_len)]
        temp.append(Is)
        temp.append(Qs)
        temp = np.transpose(temp)
        temp = MinMaxScaler(temp)
        data.append(temp)
    return data

