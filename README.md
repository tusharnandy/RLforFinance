# Reinforcement Learning in Finance
## Summer of Code '21

### Brief
-- add brief here--
### Contents
--add contents--

### A. Learning Phase
1. __Reinforcement Learning Lectures by David Silver__
	- This video series by David Silver provides a top-of-the class intro to RL 
	- The lectures are [here](https://www.davidsilver.uk/teaching/).. 
	- The pattern of topics is the same as from [this book](http://www.incompleteideas.net/book/RLbook2020.pdf) by Sutton and Barto.
	- As a fun task, I tried solving two interesting problems from it:
		- [Iterative Policy Evaluation in Small Gridworld]()
		- [Using Dynamic Programming to solve Jack's Car Rental Problem]()
2. **OpenAI SpinnigUp Blogs**
	1. [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
	2. [Types of RL algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
	3. [Introduction to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
3. **Applications of ML to Finance ([Slides](https://poseidon01.ssrn.com/delivery.php?ID=240069002121077002085006120118071075017073054032033092074008012074008113004026066069126000025041062008124018121071119092113123007080012013002115090127024028096093046036044017081025073029001025104113113070116111120067075098116101122015125110006099104&EXT=pdf&INDEX=TRUE))**

Some of the applications include:

	1. Price Prediction
	2. Hedging
	3. Portfolio Construction
	4. Risk Analysis
	5. Outlier Detection
	Apart from those mentioned above, techniques such as kernels, NLP, feature analysis, and recommendation systems have also been used lately in the world of Finance.

4. **Overview of Advanced RL in Finance ([Lectures by NYU](https://github.com/englianhu/Coursera-Overview-of-Advanced-Methods-of-Reinforcement-Learning-in-Finance))**

### B. Application Phase
1. **Following the tutorials from [pythonprogramming.net](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/)**
	1. [Implemented model free control using Q-learning on an OpenAI environment](/Q-Learning/vid1.py) 
	2. [Analysed the policy using average reward graphs](/Q-Learning/vid3.py)
	3. [Developed a custom environment from scratch and solved it using Q-learning](/Q-Learning/vid4.py)
	4. [Applied Convolution Neural Networks (CNN) as function approximator](/Q-Learning/vid6.py) (or what is know as a "[Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)")
2. **Feature extraction of stock price data (The notebook can be found here)**
	1. **Derived technical indicators from stock data such as:**
		1. Simple Moving Average (7-day and 21-day)
		2. Exponential Moving Average(12-day and 26-day)
		3. Moving Average Convergence Divergence (MACD)
		4. Bollinger Bands
		5. Schaff Trend Cycle
		6. Momentum
	2. **Used Fourier Transform for denoising the data**

		This was done by reconstructing the signal using lesser components
	3. **ARIMA as a feature**
		
		This is a technique for predicting time series data. We will show how to use it, and although ARIMA will not serve as our final prediction, we will use it as a technique to denoise the stock a little and to (possibly) extract some new patterns or features.
	4. **Autoencoder for feature extraction**
		
		Developed an encoder-decoder architecture with latent dimension = 4 to extract high-level features from the data
	5. **PCA for high-level features**
	6. **Using XGBoost for Analysing Feature Importance**
		- A benefit of using gradient boosting is that after the boosted trees are constructed, it is relatively straightforward to retrieve importance scores for each attribute.
		- Importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The performance measure may be the purity (Gini index) used to select the split points or another more specific error function.
3. 
