# linear_regression_gradient_descent
linear regression and gradient descent examples
##  Example of linear regression offer vs. demand

In this section we will perform the following step-by-step example, using the equation of maximums and minimums

![image](https://user-images.githubusercontent.com/115313115/205685034-235f718c-be34-4c04-b061-1029f80da3f3.png)

This is possible using the following lines of code
```python
#We arrange our data with the required dimensions
y = demand.reshape(demand.shape[0],1)
x = np.ones((5,2))
x[:,1] = price
#mathematics
x_transpose = np.transpose(x)
parenthesis = x_transpose.dot(x)
x_inv = np.linalg.inv(parenthesis)
x_inv_x_trans = x_inv.dot(x_transpose)
theta_pred = x_inv_x_trans.dot(y)
```
obtaining the following model:

![image](https://user-images.githubusercontent.com/115313115/205692802-fa7b8973-94e8-4ad8-a757-12fed5e4d024.png)

In this case, we evaluated the model with the MSE, obtaining an **MSE of 41.198.**

## Example linear regression with diabetes dataset from sklearn.

In this example we load our diabetes dataset from the sklearn library, perform the respective separation of the data into test and validation data, train the model and evaluate it with the MSE.

```python
#Load data
dataset, target = load_diabetes(return_X_y=True,as_frame=True)

# Add target to our data
dataset['target'] = target

# We build our dataset
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Split data for train
(x_train, x_test, y_train, y_test) = train_test_split(x,y, test_size=0.3)

#training the model
lr = LinearRegression()
reg = lr.fit(x_train,y_train)

#display the values for the function of the model
print(reg.coef_)
print(reg.intercept_)

#Predict values
val = lr.predict(x_test)
 
#MSE
error = MSE(y_test,val) 
```
Once the whole process has been carried out, we obtain the following results from the model.

![image](https://user-images.githubusercontent.com/115313115/205702125-be61bb89-0613-4eea-b3b2-1b40b4789d95.png)

## Explanation of simple gradient descent

In this example we will evaluate the gradient of a specific function in order to modify the learning rate and see how the gradient converges to the minimum of the function in the different cases, all this is possible thanks to the following function:

```python
def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=1000):
    # initialize x, f(x), and -f'(x)
    xk = x0
    fk = func(xk)
    pk = -fprime(xk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    # take steps
    while abs(pk) > tol and num_iter < max_iter:
        # calculate new x, f(x), and -f'(x)
        xk = xk + alpha * pk
        fk = func(xk)
        pk = -fprime(xk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
    # print results
    if num_iter == max_iter:
        print('Gradient descent does not converge.')
    else:
        print('Solution found:\n  y = {:.4f}\n  x = {:.4f}\n iterations = {:d}'.format(fk, xk, num_iter))
    
    return curve_x, curve_y
```

results obtained by varying the learning rate:

**$\alpha_k = 0.1$**
![image](https://user-images.githubusercontent.com/115313115/205705983-6bfe2ab0-759e-443c-9848-9418144b82da.png)
**$\alpha_k = 0.9$**
![image](https://user-images.githubusercontent.com/115313115/205706116-5e63f0f9-eb05-40d0-aac7-91fc016ad85b.png)
**$\alpha_k = 1 \times 10^{-4} $**
![image](https://user-images.githubusercontent.com/115313115/205706181-aeaf92b5-a4b0-4356-983e-2ca8eacc6333.png)
**$\alpha_k = 1.1$**
![image](https://user-images.githubusercontent.com/115313115/205706233-6205243e-3da0-411c-bd57-66fe812a04c5.png)
