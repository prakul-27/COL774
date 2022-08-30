import part1 as lr
import plotly.graph_objects as go
import numpy as np

# theta range
theta0_vals = np.linspace(0,1,100)
theta1_vals = np.linspace(-1,1,100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# compute cost for each combination of theta
xs, ys = lr.read_data()
xs, ys = lr.normalize_data(xs, ys)

c1=0; c2=0
for i in theta0_vals:
    for j in theta1_vals:        
        J_vals[c1][c2] = lr.cost(xs, ys, j, i)
        c2=c2+1
    c1=c1+1
    c2=0 # reinitialize to 0

fig = go.Figure(data=[go.Surface(x=theta0_vals, y=theta1_vals, z=J_vals)])
fig.update_layout(title='Loss function for different thetas', autosize=True,
                 xaxis_title='theta0', 
                 yaxis_title='theta1')
fig.show()