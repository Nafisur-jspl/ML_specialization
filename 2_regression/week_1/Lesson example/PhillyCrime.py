
from numpy import arange,array,ones,linalg
import pandas
import matplotlib.pyplot as plt

sourcefile = "Philadelphia_Crime_Rate_noNA.csv"
target='HousePrice'
features='CrimeRate'

sales = pandas.read_csv(sourcefile)
xi = sales[features]
A = array([xi, ones(xi.size)])
y = sales[target]
w = linalg.lstsq(A.T, y)[0]
print("Linear regression model calculated for file %s.\n------------------------\nResults:")
print("Slope:\t\t%f" %(w[0]))
print("Intercept:\t%d" %(w[1]))
print("------------------------")

#Let's paint
plt.subplot(1, 3, 1)
plt.plot(sales[features],sales[target],".",
         sales[features],w[1]+w[0]*sales[features],"g-",)
plt.xlabel('Crime Rate')
plt.ylabel('House Price ($)')
plt.title('No record filtering, data as is')
plt.grid(True)

# Filter extreme case
sales_noCC = sales[sales['MilesPhila'] != 0.0]
xi = sales_noCC[features]
A = array([xi, ones(xi.size)])
y = sales_noCC[target]
w = linalg.lstsq(A.T, y)[0]
print("Linear regression model calculated for file %s.\n------------------------\nResults:")
print("Slope:\t\t%f" %(w[0]))
print("Intercept:\t%d" %(w[1]))
print("Note: Center City records filtered")
print("------------------------")

plt.subplot(1, 3, 2)
plt.plot(sales_noCC[features],sales_noCC[target],".",
         sales_noCC[features],w[1]+w[0]*sales_noCC[features],"g-",)
plt.xlabel('Crime Rate')
plt.ylabel('House Price ($)')
plt.title('Center City records filtered')
plt.grid(True)


# Filter extreme case
sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000]
xi = sales_nohighend[features]
A = array([xi, ones(xi.size)])
y = sales_nohighend[target]
w = linalg.lstsq(A.T, y)[0]
print("Linear regression model calculated for file %s.\n------------------------\nResults:")
print("Slope:\t\t%f" %(w[0]))
print("Intercept:\t%d" %(w[1]))
print("Note: Center City and houses > 350k$ records filtered")
print("------------------------")

plt.subplot(1, 3, 3)
plt.plot(sales_nohighend[features],sales_nohighend[target],".",
         sales_nohighend[features],w[1]+w[0]*sales_nohighend[features],"g-",)
plt.xlabel('Crime Rate')
plt.ylabel('House Price ($)')
plt.title('Center City and houses > 350k$ records filtered')
plt.grid(True)


plt.show()