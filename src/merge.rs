#![allow(unused)]

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

mod iter_merge {
    use std::cmp::{Ordering, Reverse};
    use std::iter::Peekable;
    pub struct MergedIter<T: Ord, I: Iterator<Item = T>, J: Iterator<Item = T>> {
        i: Peekable<I>,
        j: Peekable<J>,
    }

    impl<T, I, J> Iterator for MergedIter<T, I, J>
    where
        T: Ord,
        I: Iterator<Item = T>,
        J: Iterator<Item = T>,
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            let oi = self.i.peek();
            let oj = self.j.peek();

            match (oi, oj) {
                (Some(si), Some(sj)) => match si.cmp(sj) {
                    Ordering::Less | Ordering::Equal => self.i.next(),
                    Ordering::Greater => self.j.next(),
                },
                (Some(_), None) => self.i.next(),
                (None, Some(_)) => self.j.next(),
                (None, None) => None,
            }
        }
    }

    /// 2つのiteratorをmergeする
    ///
    /// # Examples
    ///
    /// assert_eq!(
    ///     iter_merge(
    ///         vec![1, 4, 5, 7, 12].into_iter(),
    ///         vec![2, 3, 5, 6, 9].into_iter()
    ///     ).collect::<Vec<i32>>(),
    ///     vec![1, 2, 3, 4, 5, 5, 6, 7, 9, 12]
    /// );
    pub fn iter_merge<T, I, J>(mut i: I, mut j: J) -> MergedIter<T, I, J>
    where
        T: Ord,
        I: Iterator<Item = T>,
        J: Iterator<Item = T>,
    {
        let i = i.peekable();
        let j = j.peekable();
        MergedIter { i, j }
    }
}

/// Vecをlenによって比較するためのヘルパー構造体。
#[repr(transparent)]
pub struct SizeOrdVec<T: Ord>(pub Vec<T>);

impl<T: Ord> Ord for SizeOrdVec<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.len().cmp(&other.0.len())
    }
}

impl<T: Ord> PartialOrd for SizeOrdVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> PartialEq for SizeOrdVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
    }
}

impl<T: Ord> Eq for SizeOrdVec<T> {}

/// 2つのソート済みのSizeOrdVecをmergeして新しいSizeOrdVecを作成する。
fn merge<T: Ord>(mut vec1: SizeOrdVec<T>, mut vec2: SizeOrdVec<T>) -> SizeOrdVec<T> {
    let selfvec: Vec<T> = std::mem::replace(&mut vec1.0, Vec::default());
    let othervec: Vec<T> = std::mem::replace(&mut vec2.0, Vec::default());
    SizeOrdVec(
        iter_merge::iter_merge(selfvec.into_iter(), othervec.into_iter()).collect::<Vec<T>>(),
    )
}

/// 任意個のソート済みのSizeOrdVecをmergeする。
///
/// ReverseでSizeOrdVecを包んでいるのは高速化のため。
pub fn heap_merge<T: Ord>(mut heap: BinaryHeap<Reverse<SizeOrdVec<T>>>) -> SizeOrdVec<T> {
    if heap.len() == 0 {
        return SizeOrdVec(Vec::new());
    }

    while heap.len() > 1 {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();
        // ここでa, bはheapで最も長さが短いVecなので
        // この関数中では、mergeにかかる時間は
        // 理論上最も短くなる。
        let c = Reverse(merge(a.0, b.0));
        heap.push(c);
    }
    heap.pop().unwrap().0
}

// peekがimmutableに行えるPeekable
// その代わり初期化時点で一回はnextが呼び出される
pub struct PrefetchingPeekable<I: Iterator> {
    iter: I,
    peeked: Option<I::Item>,
}

impl<I: Iterator> PrefetchingPeekable<I> {
    #[inline]
    pub fn new(mut iter: I) -> Self {
        let peeked = iter.next();
        PrefetchingPeekable::<I> { iter, peeked }
    }

    #[inline]
    pub fn peek(&self) -> &Option<<Self as Iterator>::Item> {
        &self.peeked
    }
}

impl<I: Iterator> Iterator for PrefetchingPeekable<I> {
    type Item = I::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.peeked.take();
        self.peeked = self.iter.next();
        ret
    }
}

pub trait TPrefetchingPeekable
where
    Self: Iterator + Sized,
{
    #[inline]
    fn prefetching_peekable(self) -> PrefetchingPeekable<Self> {
        PrefetchingPeekable::new(self)
    }
}

impl<I: Iterator> TPrefetchingPeekable for I {}

// PrefetchingPeekableをself.peek()の結果で比較するラッパー構造体
#[repr(transparent)]
pub struct OrdPeekable<I>(PrefetchingPeekable<I>)
where
    I: Iterator,
    <I as Iterator>::Item: Ord;

impl<I> std::iter::Iterator for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<I> std::convert::From<I> for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    #[inline]
    fn from(from: I) -> Self {
        OrdPeekable(PrefetchingPeekable::new(from))
    }
}

impl<I> std::cmp::PartialEq for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.peek().eq(&other.0.peek())
    }
}

impl<I> std::cmp::Eq for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord + Eq,
{
}

impl<I> std::cmp::PartialOrd for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord + Eq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// peekした値を逆順で比較する
impl<I> std::cmp::Ord for OrdPeekable<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord + Eq,
{
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.peek(), other.0.peek()) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, _) => std::cmp::Ordering::Less,
            (_, None) => std::cmp::Ordering::Greater,
            (Some(a), Some(b)) => b.cmp(a),
        }
    }
}

#[repr(transparent)]
pub struct HeapMergedIter<I>(BinaryHeap<OrdPeekable<I>>)
where
    I: Iterator,
    <I as Iterator>::Item: Ord;

impl<I> HeapMergedIter<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    pub fn new() -> Self {
        Self(BinaryHeap::new())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn pop(&mut self) -> Option<OrdPeekable<I>> {
        self.0.pop()
    }

    #[inline]
    pub fn push(&mut self, iter: OrdPeekable<I>) {
        self.0.push(iter)
    }
}

impl<I> std::convert::From<BinaryHeap<OrdPeekable<I>>> for HeapMergedIter<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    fn from(from: BinaryHeap<OrdPeekable<I>>) -> Self {
        Self(from)
    }
}

impl<I> std::convert::From<HeapMergedIter<I>> for BinaryHeap<OrdPeekable<I>>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    fn from(from: HeapMergedIter<I>) -> Self {
        from.0
    }
}

impl<I> std::iter::Iterator for HeapMergedIter<I>
where
    I: Iterator,
    <I as Iterator>::Item: Ord,
{
    type Item = <I as Iterator>::Item;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            let mut iter = self.pop().unwrap();
            let ret = iter.next();
            if iter.0.peek().is_some() {
                self.push(iter);
            }
            ret
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_heap_merge() {
        let mut h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        h.push(Reverse(SizeOrdVec(vec![2, 3, 5])));
        h.push(Reverse(SizeOrdVec(vec![4, 12, 25])));
        h.push(Reverse(SizeOrdVec(vec![1, 6, 18])));
        assert_eq!(heap_merge(h).0, vec![1, 2, 3, 4, 5, 6, 12, 18, 25]);

        let mut h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        h.push(Reverse(SizeOrdVec(vec![2, 3, 5])));
        h.push(Reverse(SizeOrdVec(vec![4, 12, 25])));
        h.push(Reverse(SizeOrdVec(Vec::new())));
        assert_eq!(heap_merge(h).0, vec![2, 3, 4, 5, 12, 25]);

        let h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        assert_eq!(heap_merge(h).0, Vec::new());
    }
}
