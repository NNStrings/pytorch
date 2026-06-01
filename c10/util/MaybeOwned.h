#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace c10 {
/// MaybeOwnedTraits<T> 描述了如何从 T 进行借用。
/// 下面演示了如何使用指向常量的原始指针从任意类型 T 实现借用
/// MaybeOwnedTraitsGenericImpl抽象了【拥有】和【借用】这两个操作，类比 rust
template <typename T>
struct MaybeOwnedTraitsGenericImpl {
  using owned_type = T;     // 拥有类型就是 T 本身
  using borrow_type = const T*;   // 借用类型就是指向常对象的指针

  // 创建借用？直接取地址
  static borrow_type createBorrow(const owned_type& from) {
    return &from;
  }

  // 借用赋值？就是指针赋值
  static void assignBorrow(borrow_type& lhs, borrow_type rhs) {
    lhs = rhs;
  }

  // 销毁借用？对于裸指针，什么都不用做
  static void destroyBorrow(borrow_type& /*toDestroy*/) {}

  // 从借用获取 const 引用？直接解引用
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return *borrow;
  }

  // 从借用获取指针？直接返回指针本身
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  // 检查借用是否有效
  static bool debugBorrowIsValid(const borrow_type& borrow) {
    return borrow != nullptr;
  }
};


// 【拥有】和【借用】特征类。
// 只提供声明，不提供定义，这意味着如果你不提供特化版本就无法使用这个类
template <typename T>
struct MaybeOwnedTraits;

// 显式特化 std::shared_ptr<T>
template <typename T>
struct MaybeOwnedTraits<std::shared_ptr<T>>
    : public MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>> {};

/// 一个围绕借用或所有权的 T 的智能指针。当使用 borrowed() 构造时，
/// 调用者必须确保被借用的参数的生命周期长于这个 MaybeOwned<T>。
/// 与 Rust 的 std::borrow::Cow (https://doc.rust-lang.org/std/borrow/enum.Cow.html) 相比，
/// 但请注意，由于 C++ 没有借用检查，它可能不适用于一般用途。
/// 此处包含它是为了支持 Tensor::expect_contiguous。
template <typename T>
class MaybeOwned final {
  using borrow_type = typename MaybeOwnedTraits<T>::borrow_type;
  using owned_type = typename MaybeOwnedTraits<T>::owned_type;

  bool isBorrowed_;
  union {
    borrow_type borrow_;
    owned_type own_;
  };

  explicit MaybeOwned(const owned_type& t)
      : isBorrowed_(true), borrow_(MaybeOwnedTraits<T>::createBorrow(t)) {}

  explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
      : isBorrowed_(false), own_(std::move(t)) {}

  template <class... Args>
  explicit MaybeOwned(std::in_place_t /*unused*/, Args&&... args)
      : isBorrowed_(false), own_(std::forward<Args>(args)...) {}

 public:
  explicit MaybeOwned() : isBorrowed_(true), borrow_() {}

  MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(rhs.own_);
    }
  }

  MaybeOwned& operator=(const MaybeOwned& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = rhs.own_;
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(rhs.own_);
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  MaybeOwned(MaybeOwned&& rhs) noexcept(
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_move_assignable_v<borrow_type>)
      : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(std::move(rhs.own_));
    }
  }

  MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(
      std::is_nothrow_move_assignable_v<T> &&
      std::is_nothrow_move_assignable_v<borrow_type> &&
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_destructible_v<T> &&
      std::is_nothrow_destructible_v<borrow_type>) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = std::move(rhs.own_);
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(std::move(rhs.own_));
        isBorrowed_ = false;
      }
    }
    return *this;
  }

  static MaybeOwned borrowed(const T& t) {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(std::in_place_t /*unused*/, Args&&... args) {
    return MaybeOwned(std::in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() noexcept(
      std::is_nothrow_destructible_v<T> &&
      std::is_nothrow_destructible_v<borrow_type>) {
    if (C10_UNLIKELY(!isBorrowed_)) {
      own_.~T();
    } else {
      MaybeOwnedTraits<T>::destroyBorrow(borrow_);
    }
  }

  bool unsafeIsBorrowed() const {
    return isBorrowed_;
  }

  const T& operator*() const& {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::referenceFromBorrow(borrow_)
        : own_;
  }

  const T* operator->() const {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::pointerFromBorrow(borrow_)
        : &own_;
  }

  T operator*() && {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
      return MaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
    } else {
      return std::move(own_);
    }
  }
};

} // namespace c10
