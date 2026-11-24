//! `torch.topk(largest=True, sorted=True)` compatible selection.
//!
//! The pinned upstream (the `transformers` DiffusionGemma router) selects its
//! experts with `torch.topk` on the FP32 router probabilities. PyTorch's CPU
//! implementation is an *unstable* selection: when two experts carry exactly
//! the same probability (their BF16 logits are bit-identical), the order
//! inside that tie group is decided by the selection algorithm, not by the
//! expert index. The selected set is stable, but the order is not, and the
//! pinned checkpoint does contain such tie groups — including at the top-k
//! boundary, where the choice changes the weighted output of the model.
//!
//! To stay bit-exact with the pinned reference, this module reproduces ATen's
//! CPU algorithm exactly: `std::nth_element` (introselect) followed by
//! `std::sort` over the first `k - 1` elements, with ATen's value-only
//! comparator ("less" means a strictly greater value, NaN first). Every step
//! mirrors libstdc++'s implementation (`__introselect`, `__heap_select`,
//! `__unguarded_partition_pivot`, `__introsort_loop`, `__insertion_sort`,
//! `__final_insertion_sort`, `__adjust_heap`, ...) one for one, so the result
//! is deterministic and independent of the host toolchain's own sort
//! implementation. The emulation was checked against `torch.topk` on the 210
//! router rows of the pinned golden reference: 210/210 exact.
//!
//! The CUDA kernel `claymore_router_top_k_f32` mirrors the same algorithm;
//! the hardware differential test `hw_m10_11_router_top_k_matches_cpu`
//! checks that both paths agree.

/// The FP32 router probabilities of the first encoder row of the pinned
/// golden reference (`encoder0_router_scores.bin`, row 0 of
/// `encoder0`).
///
/// The pinned upstream's BF16 router logits of that row contain exact tie
/// groups (experts 68/33 and 22/126), one of them straddling the top-k
/// boundary. The fixture is shared by the CPU unit test and the CUDA
/// hardware differential test so both check the same selection.
pub const PINNED_ROUTER_ROW_0: [f32; 128] = [
    0.009722737,
    0.01933597,
    0.0041491548,
    0.0056270882,
    0.0038074784,
    0.0010655911,
    0.0016059024,
    0.015058869,
    0.002742424,
    0.00528616,
    0.008060432,
    0.00069204037,
    0.0009740291,
    0.006277471,
    0.00035415066,
    0.00343977,
    0.015296011,
    0.0009422193,
    0.0034939386,
    0.0024013463,
    0.0010908608,
    0.0021864509,
    0.023323633,
    0.0037484488,
    0.005990004,
    0.0030355863,
    0.006277471,
    0.005004836,
    0.004247549,
    0.001990786,
    0.0029421912,
    0.00082988315,
    0.008316297,
    0.042233646,
    0.013083371,
    0.0036048482,
    0.002721082,
    0.012880533,
    0.0068944525,
    0.0072253244,
    0.0021864509,
    0.0063763265,
    0.0011058773,
    0.0004390293,
    0.0008946926,
    0.0029192949,
    0.006277471,
    0.012290691,
    0.010512785,
    0.0031075731,
    0.008447261,
    0.004451393,
    0.0072253244,
    0.00466502,
    0.0019370881,
    0.03138538,
    0.0021694358,
    0.0010148102,
    0.0011102057,
    0.008316297,
    0.02090717,
    0.0008454251,
    0.002152553,
    0.0030593947,
    0.012484241,
    0.0010887324,
    0.0032062184,
    0.008852653,
    0.042233646,
    0.0032062184,
    0.011190801,
    0.0030593947,
    0.004486306,
    0.007572075,
    0.016538931,
    0.0043482766,
    0.0037484488,
    0.013711257,
    0.0074546807,
    0.005004836,
    0.0070030247,
    0.0033600878,
    0.0018339981,
    0.0047016083,
    0.0014339095,
    0.0043482766,
    0.0030119633,
    0.0021026891,
    0.00028792245,
    0.004416752,
    0.1222071,
    0.008187365,
    0.003928341,
    0.004486306,
    0.0033079947,
    0.0028074586,
    0.00044420443,
    0.007113306,
    0.0012506782,
    0.0045214924,
    0.003837341,
    0.0026373633,
    0.0065787327,
    0.0033079947,
    0.0025562202,
    0.021910524,
    0.0055398485,
    0.00308339,
    0.004738483,
    0.0006859847,
    0.005990004,
    0.016282521,
    0.005990004,
    0.06748917,
    0.003837341,
    0.0045926953,
    0.019036194,
    0.0053276205,
    0.006180147,
    0.002763933,
    0.0007799838,
    0.010512785,
    0.0063763265,
    0.002238301,
    0.0056270882,
    0.0045569553,
    0.023323633,
    0.0035213418,
];

/// The raw BF16 router logits of the first encoder row of the pinned
/// golden reference (`encoder0_router_raw_scores.bin`, row 0 of
/// `encoder0`), widened to FP32.
///
/// The upstream computes `softmax(logits, dtype=float32)` on those
/// logits and then selects the experts with `torch.topk`. The row
/// contains exact tie groups (experts 68/33 and 22/126), one of them
/// straddling the top-k boundary: the selection order is decided by
/// ATen's algorithm, not by the expert index. The fixture is shared by
/// the CPU unit test and the CUDA hardware differential test.
pub const PINNED_ROUTER_RAW_ROW_0: [f32; 128] = [
    2.59375,
    3.28125,
    1.7421875,
    2.046875,
    1.65625,
    0.3828125,
    0.79296875,
    3.03125,
    1.328125,
    1.984375,
    2.40625,
    -0.048828125,
    0.29296875,
    2.15625,
    -0.71875,
    1.5546875,
    3.046875,
    0.25976562,
    1.5703125,
    1.1953125,
    0.40625,
    1.1015625,
    3.46875,
    1.640625,
    2.109375,
    1.4296875,
    2.15625,
    1.9296875,
    1.765625,
    1.0078125,
    1.3984375,
    0.1328125,
    2.4375,
    4.0625,
    2.890625,
    1.6015625,
    1.3203125,
    2.875,
    2.25,
    2.296875,
    1.1015625,
    2.171875,
    0.41992188,
    -0.50390625,
    0.20800781,
    1.390625,
    2.15625,
    2.828125,
    2.671875,
    1.453125,
    2.453125,
    1.8125,
    2.296875,
    1.859375,
    0.98046875,
    3.765625,
    1.09375,
    0.33398438,
    0.42382812,
    2.4375,
    3.359375,
    0.15136719,
    1.0859375,
    1.4375,
    2.84375,
    0.40429688,
    1.484375,
    2.5,
    4.0625,
    1.484375,
    2.734375,
    1.4375,
    1.8203125,
    2.34375,
    3.125,
    1.7890625,
    1.640625,
    2.9375,
    2.328125,
    1.9296875,
    2.265625,
    1.53125,
    0.92578125,
    1.8671875,
    0.6796875,
    1.7890625,
    1.421875,
    1.0625,
    -0.92578125,
    1.8046875,
    5.125,
    2.421875,
    1.6875,
    1.8203125,
    1.515625,
    1.3515625,
    -0.4921875,
    2.28125,
    0.54296875,
    1.828125,
    1.6640625,
    1.2890625,
    2.203125,
    1.515625,
    1.2578125,
    3.40625,
    2.03125,
    1.4453125,
    1.875,
    -0.057617188,
    2.109375,
    3.109375,
    2.109375,
    4.53125,
    1.6640625,
    1.84375,
    3.265625,
    1.9921875,
    2.140625,
    1.3359375,
    0.07080078,
    2.671875,
    2.171875,
    1.125,
    2.046875,
    1.8359375,
    3.46875,
    1.578125,
];

/// A selection candidate: the probability and its expert index.
type Entry = (f32, usize);

/// ATen's comparator for `largest = true`: "less than" means a strictly
/// greater value, and NaN sorts before every number.
#[inline]
fn less(left: Entry, right: Entry) -> bool {
    (left.0.is_nan() && !right.0.is_nan()) || left.0 > right.0
}

/// `floor(log2(n))`, as libstdc++'s `__lg`.
#[inline]
fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// libstdc++ `__push_heap`.
fn push_heap(heap: &mut [Entry], mut hole: isize, top: isize, value: Entry) {
    let mut parent = (hole - 1) / 2;
    while hole > top && less(heap[parent as usize], value) {
        heap[hole as usize] = heap[parent as usize];
        hole = parent;
        parent = (hole - 1) / 2;
    }
    heap[hole as usize] = value;
}

/// libstdc++ `__adjust_heap`.
fn adjust_heap(heap: &mut [Entry], mut hole: isize, len: isize, value: Entry) {
    let top = hole;
    let mut second_child = hole;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if less(
            heap[second_child as usize],
            heap[(second_child - 1) as usize],
        ) {
            second_child -= 1;
        }
        heap[hole as usize] = heap[second_child as usize];
        hole = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        heap[hole as usize] = heap[(second_child - 1) as usize];
        hole = second_child - 1;
    }
    push_heap(heap, hole, top, value);
}

/// libstdc++ `__make_heap`.
fn make_heap(heap: &mut [Entry]) {
    let len = heap.len() as isize;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = heap[parent as usize];
        adjust_heap(heap, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// libstdc++ `__heap_select`: turns `[first, middle)` into a max-heap and
/// replaces its maximum with every greater element found in `[middle, last)`.
fn heap_select(elems: &mut [Entry], first: usize, middle: usize, last: usize) {
    let mut heap: Vec<Entry> = elems[first..middle].to_vec();
    make_heap(&mut heap);
    for entry in elems[middle..last].iter_mut() {
        if less(*entry, heap[0]) {
            let value = *entry;
            *entry = heap[0];
            adjust_heap(&mut heap, 0, (middle - first) as isize, value);
        }
    }
    elems[first..middle].copy_from_slice(&heap);
}

/// libstdc++ `__partial_sort`: `__heap_select` followed by `__sort_heap` over
/// `[first, middle)`.
fn partial_sort(elems: &mut [Entry], first: usize, middle: usize, last: usize) {
    heap_select(elems, first, middle, last);
    let mut heap: Vec<Entry> = elems[first..middle].to_vec();
    let mut length = heap.len();
    while length > 1 {
        length -= 1;
        let value = heap[length];
        heap[length] = heap[0];
        adjust_heap(&mut heap, 0, length as isize, value);
    }
    elems[first..middle].copy_from_slice(&heap);
}

/// libstdc++ `__move_median_to_first`.
fn move_median_to_first(elems: &mut [Entry], result: usize, a: usize, b: usize, c: usize) {
    if less(elems[a], elems[b]) {
        if less(elems[b], elems[c]) {
            elems.swap(result, b);
        } else if less(elems[a], elems[c]) {
            elems.swap(result, c);
        } else {
            elems.swap(result, a);
        }
    } else if less(elems[a], elems[c]) {
        elems.swap(result, a);
    } else if less(elems[b], elems[c]) {
        elems.swap(result, c);
    } else {
        elems.swap(result, b);
    }
}

/// libstdc++ `__unguarded_partition`.
fn unguarded_partition(
    elems: &mut [Entry],
    mut first: isize,
    mut last: isize,
    pivot_index: usize,
) -> isize {
    let pivot = elems[pivot_index];
    loop {
        while less(elems[first as usize], pivot) {
            first += 1;
        }
        last -= 1;
        while less(pivot, elems[last as usize]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        elems.swap(first as usize, last as usize);
        first += 1;
    }
}

/// libstdc++ `__unguarded_partition_pivot`.
fn unguarded_partition_pivot(elems: &mut [Entry], first: isize, last: isize) -> isize {
    let mid = first + (last - first) / 2;
    move_median_to_first(
        elems,
        first as usize,
        (first + 1) as usize,
        mid as usize,
        (last - 1) as usize,
    );
    unguarded_partition(elems, first + 1, last, first as usize)
}

/// libstdc++ `__unguarded_linear_insert`.
fn unguarded_linear_insert(elems: &mut [Entry], mut last: usize) {
    let value = elems[last];
    let mut previous = last - 1;
    while less(value, elems[previous]) {
        elems[last] = elems[previous];
        last = previous;
        previous -= 1;
    }
    elems[last] = value;
}

/// libstdc++ `__insertion_sort`.
fn insertion_sort(elems: &mut [Entry], first: usize, last: usize) {
    if first == last {
        return;
    }
    for index in (first + 1)..last {
        if less(elems[index], elems[first]) {
            let value = elems[index];
            elems.copy_within(first..index, first + 1);
            elems[first] = value;
        } else {
            unguarded_linear_insert(elems, index);
        }
    }
}

/// libstdc++ `__unguarded_insertion_sort`.
fn unguarded_insertion_sort(elems: &mut [Entry], first: usize, last: usize) {
    for index in first..last {
        if less(elems[index], elems[first]) {
            let value = elems[index];
            elems.copy_within(first..index, first + 1);
            elems[first] = value;
        } else {
            unguarded_linear_insert(elems, index);
        }
    }
}

/// libstdc++ `__introsort_loop`.
fn introsort_loop(elems: &mut [Entry], first: usize, mut last: usize, mut depth: u32) {
    const THRESHOLD: usize = 16;
    while last - first > THRESHOLD {
        if depth == 0 {
            partial_sort(elems, first, last, last);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(elems, first as isize, last as isize) as usize;
        introsort_loop(elems, cut, last, depth);
        last = cut;
    }
}

/// libstdc++ `__final_insertion_sort`.
fn final_insertion_sort(elems: &mut [Entry], first: usize, last: usize) {
    const THRESHOLD: usize = 16;
    if last - first > THRESHOLD {
        insertion_sort(elems, first, first + THRESHOLD);
        unguarded_insertion_sort(elems, first + THRESHOLD, last);
    } else {
        insertion_sort(elems, first, last);
    }
}

/// libstdc++ `std::sort`.
fn std_sort(elems: &mut [Entry], first: usize, last: usize) {
    if first != last {
        introsort_loop(elems, first, last, (lg(last - first) * 2) as u32);
        final_insertion_sort(elems, first, last);
    }
}

/// libstdc++ `__introselect`.
fn introselect(elems: &mut [Entry], mut first: usize, nth: usize, mut last: usize, mut depth: u32) {
    while last - first > 3 {
        if depth == 0 {
            // libstdc++ calls `__heap_select(first, nth + 1, last)` and then
            // swaps `first` and `nth` to place the nth largest element in its
            // final position.
            heap_select(elems, first, nth + 1, last);
            elems.swap(first, nth);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(elems, first as isize, last as isize) as usize;
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(elems, first, last);
}

/// libstdc++ `std::nth_element`.
fn nth_element(elems: &mut [Entry], first: usize, nth: usize, last: usize) {
    if first == last || nth == last {
        return;
    }
    introselect(elems, first, nth, last, (lg(last - first) * 2) as u32);
}

/// Select the `k` largest values, in the exact order returned by
/// `torch.topk(values, k, largest=True, sorted=True)` on CPU.
///
/// The returned indices are the expert indices, ordered by decreasing
/// probability. Ties keep the order produced by ATen's unstable selection.
pub fn top_k_largest(values: &[f32], k: usize) -> Vec<usize> {
    let mut entries: Vec<Entry> = values.iter().copied().zip(0..).collect();
    if k == 0 || entries.is_empty() {
        return Vec::new();
    }
    let k = k.min(entries.len());
    let count = entries.len();
    nth_element(&mut entries, 0, k - 1, count);
    std_sort(&mut entries, 0, k - 1);
    entries[..k].iter().map(|(_, index)| *index).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned upstream's selection for the first encoder router row of the
    /// golden eager reference (`encoder0_router_indices`), whose BF16 logits
    /// contain two exact tie groups (experts 68/33 and 22/126).
    #[test]
    fn reproduces_the_pinned_upstream_tie_order() {
        // The 128 FP32 router probabilities of `encoder0_router_scores.bin`,
        // first row, exactly as the pinned reference produced them.
        let probabilities = PINNED_ROUTER_ROW_0;
        assert_eq!(probabilities.len(), 128);
        assert_eq!(
            top_k_largest(&probabilities, 8),
            vec![90, 113, 68, 33, 55, 22, 126, 105]
        );
    }

    /// The same selection, computed from the raw BF16 logits: the softmax
    /// preserves the exact ties, so the ATen algorithm still decides the order
    /// of the tied group (and of the group straddling the top-k boundary).
    #[test]
    fn reproduces_the_pinned_upstream_tie_order_from_the_raw_logits() {
        let logits = PINNED_ROUTER_RAW_ROW_0;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
        let sum: f32 = probabilities.iter().sum();
        for probability in probabilities.iter_mut() {
            *probability /= sum;
        }
        assert_eq!(
            top_k_largest(&probabilities, 8),
            vec![90, 113, 68, 33, 55, 22, 126, 105]
        );
    }

    #[test]
    fn selects_the_largest_values() {
        let values = [0.5f32, 0.25, 0.75, 0.125, 1.0, 0.625];
        assert_eq!(top_k_largest(&values, 3), vec![4, 2, 5]);
        assert_eq!(top_k_largest(&values, 0), Vec::<usize>::new());
        assert_eq!(top_k_largest(&values, 99).len(), values.len());
    }

    #[test]
    fn keeps_every_expert_of_a_tied_group() {
        let values = [1.0f32, 1.0, 1.0, 0.5];
        let selected = top_k_largest(&values, 3);
        assert_eq!(selected.len(), 3);
        for index in selected {
            assert!(index < 3, "tied expert {index} must stay in the selection");
        }
    }
}
