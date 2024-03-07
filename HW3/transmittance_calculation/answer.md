# Answers

- NOTE: T(x,y1) = 1 (assumed)

```
T(y1, y2) = T(y1,y1) * e^(-sigma_delta_t * delta_t)
          = 1 * e^(-1 * 2)
          = 0.1353
```

```
T(y2, y4) = T(y1, y3) * e^(-sigma_delta_t * delta_t)
          = [ T(y1, y2) * e^(-sigma_delta_t * delta_t) ] * e^(-sigma_delta_t * delta_t)
          = [0.1353 * 0.60653] * 0
          = 0

          NOTE: T(y1, y3) = [ T(y1, y2) * e^(-sigma_delta_t * delta_t) ] = 
```

```
T(x, y4) = 0
```

```
T(x, y3) = T(x, y1) * T(y1, y2) * T(y2, y3)
         = T(x, y1) * T(y1, y3)
         = 1 * 0.0820
         = 0.0820
```