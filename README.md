# DWHRec
Against Filter Bubbles: Diversified Music Recommendation via Weighted Hypergraph Embedding Learning
***
## Description
The code for the paper titled 'Against Filter Bubbles: Diversified Music Recommendation via Weighted Hypergraph Embedding Learning' is abbreviated as **DWHRec**.

##  Structure
The project is organized into two principal directories: `src` and `datasets`.

The `src` directory houses the implementation of the algorithms, whereas the `datasets` directory encompasses the datasets employed for testing.

The file `main.py`, situated at the same directory level as `src` and `datasets`, functions as the entry point for the program.

## Instruction
`main.py` contains some parameters.

| Parameters | Meanings | Default value |
|:---:|:---:| :---: |
| --dataset | The specified name indicates the dataset to be loaded. | 100k |
| --r | The variable $r$ denotes the number of iterations performed in the random walk process. | 5 |
| --k | The variable $k$ specifies the number of steps taken during each iteration of the random walk. | 100 |
| --s | The symbol $s$ represents the dimensional space in which the vector representations of users and items are defined. | 50 |
| --w | The parameter $w$ signifies the extent of the context window within the skip-gram model. | 5 |

## Usage
Execute the subsequent command within any terminal or terminal-emulating application.

For example:

(1). A straightforward approach to utilizing the system is as follows:
```
python main.py
```

In this instance, all parameters have been assigned their default values.

(2). 
If you desire to assign custom values to each parameter, please follow the procedure below.
```
python main.py --dataset 100k --r 5 --k 200 --s 200 --w 5
```
