# hydroelectricity_forecast
This project has begun because of the need to forecast hydroelectricity consumption. As part of my research, I was tasked with analyzing the time series data of hydroelectricity consumption and developing a model to accurately forecast its future trends. To tackle this problem, I foremost imported various libraries such as numpy, pandas, and matplotlib.pyplot to analyze the data.

First, I visualized the time series data, which helped me understand the consumption trends. I plotted the production graph, which showed me how the consumption varied with time. I also plotted the kernel density estimate plot, which helped me identify the peaks in consumption.

Next, I tried to stationarize the time series data by using the Dickey-Fuller test. This test allowed me to identify whether the data had a trend, which could be removed to make the time series stationary. I also used the rolling mean and standard deviation to better understand the data.

After that, I applied a logarithmic transformation to the data to stabilize the variance, which was needed to stationarize the data. This helped me to see the changes in consumption over time without the influence of any trends. I used the rolling mean and standard deviation again to observe the changes in the data.

I also used the exponentially weighted moving average to smoothen the data and make it easier to see the patterns. This technique allowed me to identify any seasonality in the data and remove it.

Finally, I used auto-correlation and partial auto-correlation plots to determine the best parameters for my model. These plots helped me to identify the degree of correlation between the consumption at different points in time.

In conclusion, by analyzing the time series data and stationarizing it, I was able to better understand the trends and patterns in hydroelectricity consumption. I used various techniques to remove trends, smoothen the data, and identify the best parameters for my model.
