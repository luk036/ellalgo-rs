Python

- garbage collection
- primitive types (bool, int, float, tuple)- call by value
- object types (list, dict, ...)- call by reference
- All objects are optional (None)

```python
def update(tea):
    if condition:
        return None  # no update
    ...
    tea += ... # update tea
    return tea
```

C++

```cpp
template <typename T>
auto update(T& tea) -> bool {
    if (condition) {
        return false;  // no update
    }
    tea += ...; // update tea
    return true;
}
```

Rust:

```rust
fn update(tea: &mut T) -> bool {
    if condition {
        return false;
    }
    tea += ... // update tea 
    true
}
```

```python
def solve(ss):
    x_best = None
    for niter in range(1000):
        ...
        if found:
            x_best = ss.xc.copy()
    ...
    return x_best

x = solve(ss)
if x:
    process(x)
```

C++

```cpp
template <typename SS>
auto solve(SS& ss) -> std::optional<Arr> {
    auto x_best = std::optional<Arr>{};
    for (auto niter=0; niter < 1000; ++niter) {
        if (found) {
            x_best = ss.xc();
        }
    }
    return x_best;
}

auto x_opt = solve(ss);
if (x_opt) {
   process(*x_opt);
} 
```

Rust:

```rust
fn solve(ss: &mut SS) -> Option<Arr> {
    let x_best = None
    for niter in 0..1000 {
        if (found) {
            x_best = Some(ss.xc());
        }
    }
    return x_best;
}

let x_opt = solve(ss);
if let Some(x) = x_opt {
    process(x);
}
```
