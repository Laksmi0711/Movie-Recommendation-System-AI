# Movie Recommendation System

### Overview
Our group has developed an Movie Recommendation System using Python Streamlit. This project demonstrates how machine learning principles and web development tools can be combined to create a user-friendly application for product recommendations.

Dataset: https://grouplens.org/datasets/movielens/

Website Link: https://e-commercerecomsystem-hlm3dhcxb8thcblmtedmay.streamlit.app/

---
### Installation and Configuration
**Step 1: Install Streamlit**
```bash
pip install streamlit  
```
**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```
**Step 3: Prepare the Dataset, Model and Python Script**
```
<your-repo-name>/
├── ml-1m/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── model/
│   └── movie_recommendation_model.pth
├── streamlit_app.py
├── utils.py
├── requirements.txt
└── README.md
```
**Step 4: Run the App**
```bash
streamlit run app.py
```

### Deployment
This project can be deployed on platforms like [Render](https://dashboard.render.com/) and [Streamlit Community Cloud](https://streamlit.io/cloud). Follow the respective platform’s guidelines for deployment. Ensure the following files are included:
- ```app.py```
- ```requirements.txt```

---

### Reference
#### 1. App Deployment Tutorial
- [Dash Python App Deployment Tutorial](https://www.youtube.com/watch?v=H16dZMYmvqo&t=142s)
- [How to Deploy Your App to Streamlit Community Cloud](https://www.youtube.com/watch?v=HKoOBiAaHGg&t=427s)
#### 2. Recommendation System
- [Mastering Recommendation System: A Complete Guide](https://learnopencv.com/recommendation-system/)
- [Deep Learning Recommendation Models (DLRM): A Deep Dive](https://medium.com/swlh/deep-learning-recommendation-models-dlrm-a-deep-dive-f38a95f47c2c)
- [Practical Guide to Building Scalable Recommender Systems in Python](https://medium.com/@anilcogalan/practical-guide-to-building-scalable-recommender-systems-in-python-b175547e6fce)
- [How to Build a Winning Recommendation System, Part 1](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-1/)
- [How to Build a Deep Learning Powered Recommender System, Part 2](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-2-deep-learning-for-recommender-systems/)
- [How to Build a Winning Deep Learning Recommender System (Youtube Video)](https://www.youtube.com/watch?v=bHuww-l_Sq0)
