# RAS Digitalization Project

This is project was done in collaboration with [Billund Aquaculture A/S](https://www.billundaquaculture.com/).  

It presents a Forecasting AI trained to predict the total output for the indoor recirculating aquaculture systems, based on real data coming from sensors with adaptable behavior for missing inputs. The prediction is in a sequential manner, using  (Long Short-Term Memory) neural networks as the main building block for the AI, with time-based data as inputs.

## Installation

A Python version newer than 3.8 is recommended. (Preferably Python 3.10)

Install the required packages using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

Also for experiencing the full capabilities of the project, there might be a need to have an Nvidia Graphics Card with CUDA compatibility. More details about it can be found [here](https://developer.nvidia.com/cuda-gpus)

## Usage

```bash
# For starting the training 
cd src
python train.py
```

## Missing 
Due to non-disclosure agreements, some of the code contents and the data are missing from the repository.

## Contact

For further discussions, ideas, or collaborations please contact: [manuotel@gmail.com](manuotel@gmail.com)