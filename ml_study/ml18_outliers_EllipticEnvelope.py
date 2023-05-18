import numpy as np

oliers = np.array([-50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

oliers = oliers.reshape(-1, 1)
print(oliers.shape) #(14, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)
outliers.fit(oliers)
result = outliers.predict(oliers)

print(result)
print(result.shape)
