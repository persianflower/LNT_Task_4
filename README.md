
# Task 4: Developing a Flask API for Deep Learning Models

### Objective
To expose trained deep learning models as REST APIs using the Flask framework.

### Deliverables

1. Flask application source code.
   
   [Flask code](https://github.com/persianflower/LNT_Task_4/blob/main/main.py)
   
2. API documentation.
   
   [API documentation](https://github.com/persianflower/LNT_Task_4/blob/main/api.py)
  
3. Sample prediction outputs.

![Class labels](https://github.com/persianflower/LNT_Task_4/blob/main/Labels.png)
   
  - Input:
  
    Ankle_boot
    
![Processed input image](https://github.com/persianflower/LNT_Task_4/blob/main/ankle_thumb.png)

    Output:
    
![Output](https://github.com/persianflower/LNT_Task_4/blob/main/ankle_op.png)

  - Input:

    Shirt
    
![Processed input image](https://github.com/persianflower/LNT_Task_4/blob/main/shirt_thumb.png)

    Output:
    
 ![Output](https://github.com/persianflower/LNT_Task_4/blob/main/shirt_op.png)
 

  - Input:

    Sandal
    
![Processed input image](https://github.com/persianflower/LNT_Task_4/blob/main/sandal_thumb.png)

    Output:
    
![Output](https://github.com/persianflower/LNT_Task_4/blob/main/sandal_op.png)


### Result

| Predicted Output | Actual Output |
| -- | -- |
| Class = 9 | Class = 9 |
| Class = 6 | Class = 6 |
| Class = 8 | Class = 5 |

### Observation

From the above testing of our training model various things can be concluded. First, since the model uses Fashion MNIST dataset the custom input images also were required to be 
preprocessed in the same way. This processing was done with the help of the paper on the [MNIST dataset](https://arxiv.org/pdf/1708.07747). When we talk about the accuracy of our data,
while training we observed that our validation accuracy was (~90%), this is proven true with the help of the custom input testing that we carried out. As can be observed via the result
table above. Two of the three tested classes was correctly identified by our model, while one was wrong (Sandal predicted as bag). For future purposes this model could be trained for 
more epochs to increase its accuracy. Alternatively, we could find all the classes which it incorrectly identifies and further train it on such images. 

