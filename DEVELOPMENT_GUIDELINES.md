# Development Guidelines

This document contains important lessons learned and best practices for developing the AV stack.

## Critical Lessons Learned

### 1. Test Empirically, Don't Just Reason Theoretically

**The Problem:**
We added a steering direction fix (negation), then removed it based on theoretical reasoning about sign conventions, then had to add it back when data proved it was needed. This wasted time and caused confusion.

**The Lesson:**
- **Always write tests FIRST** for critical control logic (steering direction, error correction, etc.)
- **Test with known scenarios** - don't just reason about sign conventions
- **Verify empirically** - run the test and see what happens
- **Don't remove fixes** based on theoretical reasoning alone - verify with data

**Example:**
```python
# ✅ GOOD: Test steering direction empirically
def test_lateral_controller_steering_direction():
    """Verify car steers TOWARD reference, not away."""
    # Car RIGHT of ref → should steer LEFT (negative)
    # Car LEFT of ref → should steer RIGHT (positive)
    # Test with actual positions and verify direction
```

**What We Did:**
- Created `test_lateral_controller_steering_direction()` to catch this issue
- Added validation logging in `av_stack.py` to warn if steering direction is wrong
- Documented this lesson here

---

### 2. Sign Convention Ambiguity

**The Problem:**
Coordinate system conventions (Unity vs Vehicle vs Image) can be ambiguous. What does "positive X" mean? What does "negative steering" mean?

**The Lesson:**
- **Document coordinate systems** clearly in code comments
- **Test with concrete examples** - don't assume conventions
- **Use consistent naming** - make it clear what positive/negative means
- **Add validation** - check that signs match expectations

**Example:**
```python
# ✅ GOOD: Document coordinate system
# Unity coordinates: +X = RIGHT, +Z = FORWARD
# Vehicle coordinates: +X = RIGHT, +Y = FORWARD
# Steering: positive = steer RIGHT, negative = steer LEFT
```

---

### 3. Control Logic Requires Testing

**The Problem:**
Control systems (PID controllers, steering, speed control) are complex and sign conventions matter. Bugs in control logic can cause dangerous behavior.

**The Lesson:**
- **Write unit tests** for all control logic
- **Test edge cases** - what happens at limits? What happens with zero error?
- **Test direction** - verify corrections go in the right direction
- **Test integration** - verify components work together correctly

**Checklist for Control Logic:**
- [ ] Unit test for direction (does it correct errors correctly?)
- [ ] Unit test for limits (does it saturate correctly?)
- [ ] Unit test for edge cases (zero error, extreme errors)
- [ ] Integration test (does it work with real data?)
- [ ] Validation logging (warn if something looks wrong)

---

## Testing Best Practices

### When to Write Tests

**ALWAYS write tests for:**
1. **Control logic** - steering, speed, PID controllers
2. **Coordinate transformations** - image to vehicle, world to vehicle
3. **Error calculations** - lateral error, heading error
4. **Critical algorithms** - trajectory planning, lane detection
5. **Configuration loading** - ensure parameters are loaded correctly

**Write tests BEFORE fixing bugs:**
1. Reproduce the bug in a test
2. Fix the bug
3. Verify the test passes
4. This ensures the bug doesn't come back

### Test Structure

```python
def test_feature_name():
    """
    Test description.
    
    What it tests and why it's important.
    """
    # Setup
    controller = SomeController()
    
    # Test case 1: Normal operation
    result = controller.compute(...)
    assert result == expected
    
    # Test case 2: Edge case
    result = controller.compute(...)
    assert result == expected
    
    # Test case 3: Error case
    with pytest.raises(SomeError):
        controller.compute(...)
```

---

## Code Review Checklist

Before committing control logic changes:

- [ ] **Tests written** - Unit tests for new/changed logic
- [ ] **Tests pass** - All tests pass locally
- [ ] **Direction verified** - For control logic, verify direction is correct
- [ ] **Edge cases handled** - Test with zero, extreme, and boundary values
- [ ] **Documentation updated** - Code comments explain coordinate systems and conventions
- [ ] **Validation added** - Runtime checks for common mistakes

---

## Common Pitfalls

### 1. Sign Convention Confusion
**Symptom:** Control works in simulation but wrong direction in real data  
**Prevention:** Write test with known positions, verify direction

### 2. Theoretical Reasoning Without Testing
**Symptom:** "This should work" but it doesn't  
**Prevention:** Write test first, verify empirically

### 3. Removing Fixes Based on Theory
**Symptom:** Fix works, but removed because "it shouldn't be needed"  
**Prevention:** Keep fix if data shows it's needed, even if theory says otherwise

### 4. Coordinate System Assumptions
**Symptom:** Works in one scenario, fails in another  
**Prevention:** Document coordinate systems, test transformations

---

## Debugging Workflow

When debugging control issues:

1. **Reproduce in test** - Create minimal test case
2. **Check data** - Analyze recordings/logs
3. **Verify direction** - Does correction go in right direction?
4. **Check signs** - Are coordinate systems consistent?
5. **Add logging** - Log intermediate values
6. **Fix and verify** - Fix bug, verify test passes

---

## References

- **Steering Direction Fix:** See `tests/test_control.py::test_lateral_controller_steering_direction`
- **Validation Logging:** See `av_stack.py` (steering direction warnings)
- **Coordinate Systems:** See `control/pid_controller.py` (lateral error calculation)

---

## Remember

> **"Test empirically, not theoretically. Data beats reasoning."**

When in doubt, write a test. When fixing a bug, write a test first. When reasoning about signs, test with concrete examples.

