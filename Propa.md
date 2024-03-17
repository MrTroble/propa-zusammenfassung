# Propa

## Haskell

> Definitionen Funktion f

```haskell
f x = sin x / x -- 1 parameter function
f a x = a * x * x -- 2 parameter function
```

> Fallunterscheidung if

```haskell
binom n k =
    if (k==0) || (k==n)
    then 1
    else binom (n-1) (k-1) + binom (n-1) k
```

> Fallunterscheidung ohne if

```haskell
binom n k
    |(k==0 || k==n) = 1
    |otherwise = binom (n-1) (k-1) + binom (n-1) k
```

> Fallunterscheidung Pattern Matching

```haskell
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
```

> `Tail recursion` Eine Funktion heißt linear rekursiv, wenn in jedem Definitionszweig nur ein rekursiver Aufruf vorkommt. Eine linear rekursive Funktion heißt tail recursive, wenn in jedem Zweig der rekursive Aufruf nicht in andere Aufrufe eingebettet ist.

### Lists/Tuple

```haskell
(x:xs) -- Cons operator (create list from head and rest)
[] -- Empty List
(x:[]) -- One list element
[1, 2, 3, 4] -- Create list with 4 elements
(1, 2, 3, 4) -- Create tuple with 4 elements
head l -- First element of List (x)
tail l -- Without the first element l (xs)
take n l -- First n elements from l
drop n l -- l without first n elements
app a b -- Append two lists
a ++ b -- Infix notation off app
length l -- length of the list
concat [l1, l2, l3] -- Flattens elements of l to one list
filter pre l -- Filter list with predicate
map f l -- Maps each element to an other with the function
zipWith f l1 l2 -- 
```

> Foldr/Foldl

```haskell
foldr op i [] = i
foldr op i (x:xs) = op x (foldr op i xs)

foldl op i [] = i
foldl op i (x:xs) = foldl op (op i x) xs
```

### Lambdas/Bindings

```haskell
f . g -- Composition infix notation
\x -> sin x / x -- Lambda abstraction
-- Unterversorgung:
(+) -- Infix (addition/...) as function parameter 
(+1) -- Parameterized infix notation as function
([]++) -- Parameterized list infix as function
(,) -- Tuple creation as function
(:) -- Cons as a function
-- Bind c in let and use in in body
f x = let c = 200
      in x * c
-- Bind c in where clause and use before
f x = x * c
      where c = 200
-- More power then let as it allows self recursion
```

> WHITESPACE SENSITIVE!
