#pragma once

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>
#include <atomic>
#include <climits>
#include <memory>
#include <type_traits>

namespace pybind11 {
template <typename, typename...>
class class_;
}

namespace torch::utils {
class PyObjectPreservation;
}

namespace c10 {
class intrusive_ptr_target;
namespace raw {
namespace weak_intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}
namespace intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
}

// 被 intrusive_ptr 构造函数使用的构造函数标签，构造时不增加引用计数
struct DontIncreaseRefcount {};
} // namespace raw

namespace detail {
constexpr uint64_t kImpracticallyHugeReferenceCount = 0x0FFFFFFF;
constexpr uint64_t kImpracticallyHugeWeakReferenceCount =
    (kImpracticallyHugeReferenceCount << 32);
constexpr uint64_t kReferenceCountOne = 1;
constexpr uint64_t kWeakReferenceCountOne = (kReferenceCountOne << 32);
constexpr uint64_t kUniqueRef = (kReferenceCountOne | kWeakReferenceCountOne);
// 表明对象是否有一个 PyObject 包装器
constexpr uint64_t kHasPyObject = (uint64_t(1) << 63);

// 指针的空类型，默认为 nullptr
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;
  }
};

template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}

inline uint32_t refcount(uint64_t combined_refcount) {
  return static_cast<uint32_t>(combined_refcount);
}

inline uint32_t weakcount(uint64_t combined_refcount) {
  return static_cast<uint32_t>((combined_refcount & ~kHasPyObject) >> 32);
}

inline bool has_pyobject(uint64_t combined_refcount) {
  return (combined_refcount & kHasPyObject) != 0;
}

inline bool is_uniquely_owned(uint64_t combined_refcount) {
  return (combined_refcount & ~detail::kHasPyObject) == detail::kUniqueRef;
}

// 对引用计数递增的唯一要求是它发生在递减之前，因此不需要额外的内存顺序。
inline uint64_t atomic_combined_refcount_increment(
    std::atomic<uint64_t>& combined_refcount,
    uint64_t inc) {
  return combined_refcount.fetch_add(inc, std::memory_order_relaxed) + inc;
}

inline uint32_t atomic_weakcount_increment(
    std::atomic<uint64_t>& combined_refcount) {
  return detail::weakcount(atomic_combined_refcount_increment(
      combined_refcount, kWeakReferenceCountOne));
}

// 要求是所有对托管对象的修改都发生在托管对象析构函数调用之前，
// 并且托管对象存储的分配发生在存储释放之前。
//
// 为了获得这种顺序，所有非最终的递减必须与最终的递减同步。
// 因此，所有非最终的递减必须使用 store-release，而最终的递减必须使用 load-acquire，
// 要么直接使用，要么借助内存屏障。但最简单的方法是让所有递减都使用 acq-rel。
// 事实证明，在现代架构和芯片上，这也是最快的。
inline uint64_t atomic_combined_refcount_decrement(
    std::atomic<uint64_t>& combined_refcount,
    uint64_t dec) {
  return combined_refcount.fetch_sub(dec, std::memory_order_acq_rel) - dec;
}

inline uint32_t atomic_weakcount_decrement(
    std::atomic<uint64_t>& combined_refcount) {
  return detail::weakcount(atomic_combined_refcount_decrement(
      combined_refcount, kWeakReferenceCountOne));
}

// 判断 T 是否等于 c10::intrusive_ptr_target
template <class T, class = void>
struct TargetTraits {
  static constexpr bool can_have_pyobject = false;
};

} // namespace detail

/**
 * intrusive_ptr<T> 是 shared_ptr<T> 的一种替代方案，采用侵入式引用计数，
 * 你的类 T 需要继承自 intrusive_ptr_target 才能被 intrusive_ptr<T> 使用。
 * 你的类的构造函数不应该允许 `this` 逃逸到其他线程，也不应该从 `this` 创建 intrusive_ptr。
 */

// 注意 [栈上分配的 intrusive_ptr_target 的安全性]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// std::enable_shared_from_this 的一个众所周知的问题是，它允许你从栈上分配的对象
// 创建 std::shared_ptr，这完全是荒谬的，因为一旦从栈返回，该对象就会被销毁。
// 在 intrusive_ptr 中，我们可以检测到这种情况的发生，因为我们将继承自
// intrusive_ptr_target 的对象的引用计数/弱引用计数设置为零，*除非* 我们可以证明
// 该对象是动态分配的（例如，通过 make_intrusive）。
//
// 因此，每当你将 T* 转换为 intrusive_ptr<T> 时，我们会检查并确保引用计数不为零
// （或者对于 weak_intrusive_ptr<T> 来说，有一个更微妙的测试：引用计数可能有效为零，
// 但弱引用计数最好不要为零），因为这告诉我们是是否由我们自己分配了该对象。
// 如果不是，就不能给你 intrusive_ptr！

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class C10_API intrusive_ptr_target {
  // 注意 [侵入式引用计数的弱引用]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 方案如下：
  //
  //  - refcount == 指向对象的强引用数量
  //    weakcount == 指向对象的弱引用数量，
  //      如果 refcount > 0，则 weakcount 再加一
  //    不变式：refcount > 0  =>  weakcount > 0
  //
  //  - 只要有指向 c10::StorageImpl 的强指针或弱指针存在（weakcount > 0，
  //    因为强引用会使得 weakcount 加一），c10::StorageImpl 就会保持存活
  //
  //  - 当 refcount == 0 时，会调用终结器，并且 data_ptr 会被释放
  //
  //  - 一旦 refcount == 0，就再也不可能 > 0（从 > 0 到 == 0 的转变是单调的）
  //
  //  - 当你通过弱指针访问 c10::StorageImpl 时，如果引用计数大于 0，
  //    你必须原子地增加使用计数。如果它不大于 0，你必须报告该存储已经死亡。
  //
  // 我们使用一个组合计数来同时表示 refcount 和 weakcount，
  // 这样我们就可以原子地同时操作两者，以获得更好的性能和确定的行为。

  // combined_refcount_ 格式：低 32 位是 refcount，33-63 位是 weakcount，最高位是 PyObject 标识
  // 注意 [Tensor 和 Storage 的 PyObject 保留]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // intrusive_ptr 对 TensorImpl 和 StorageImpl 的 PyObject 包装器提供了特殊支持。
  // 组合引用计数 combined_refcount_ 的最高有效位（kHasPyObject）用于指示该对象是否拥有 PyObject 包装器。
  //
  //   - 如果 PyObject 存在，它将持有一个对 intrusive_ptr_target 的强引用。
  //
  //   - 当 refcount 从 1 变为 2 时，我们会增加 PyObject 的引用计数（incref）。
  //
  //   - 当 refcount 从 2 变为 1 时，我们会减少 PyObject 的引用计数（decref）。
  //
  // 换句话说，只要存在其他指向 intrusive_ptr_target 的 C++ 引用，
  // intrusive_ptr 就会保持 PyObject 存活。

  mutable std::atomic<uint64_t> combined_refcount_;
  static_assert(sizeof(std::atomic<uint64_t>) == 8);
  static_assert(alignof(std::atomic<uint64_t>) == 8);
  // is_always_lock_free 的检查位于 intrusive_ptr.cpp 中，因此它只由
  // 主机编译器进行评估。类似 CUDA 的设备编译器在解析此头文件时可能会
  // 针对没有 64 位原子操作的硬件进行编译；请参见
  // https://github.com/pytorch/pytorch/issues/171775（由
  // https://github.com/pytorch/pytorch/pull/163394 引入）。

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);

  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;
  friend inline void raw::weak_intrusive_ptr::incref(
      intrusive_ptr_target* self);

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;

  friend class torch::utils::PyObjectPreservation;

 protected:
  // protected 析构函数。我们永远不希望直接析构 intrusive_ptr_target*。
  virtual ~intrusive_ptr_target() {}

  constexpr intrusive_ptr_target() noexcept : combined_refcount_(0) {}

  // intrusive_ptr_target 支持 copy 和 move：但 refcount 和 weakcount 不参与
  // （因为它们是 memory location 的内在属性）
  // 所以赋值和移动构造也等同于默认构造，不会真的赋值
  intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
    return *this;
  }

  intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(
      const intrusive_ptr_target& /*other*/) noexcept {
    return *this;
  }

 private:
  /**
   * 当 refcount 变为零时调用此函数。
   * 你可以重写此函数来释放昂贵的资源。
   * 此时可能仍然存在 weak references，因此你的对象可能尚未被析构，
   * 但你可以假定该对象不再被使用，即不会再调用其方法或访问其成员
   * （我们之所以还不能析构它，是因为需要保证 weakcount 可访问）。
   *
   * 如果不存在 weak references（即你的对象即将被析构），
   * 则不会调用此函数。
   */
  virtual void release_resources() {}

  /**
   * 当 refcount 在 1 和 2 之间转换并且对象有 PyObject wrapper 时，
   * 会调用这两个方法。
   */
  virtual void incref_pyobject() const noexcept {}
  virtual void decref_pyobject() const noexcept {}
  virtual bool try_incref_pyobject() const noexcept {
    return false;
  }

  // 这里的封装将数据转换成 uint32_t，并处理最高有效位（kHasPyObject）
  uint32_t refcount(std::memory_order order = std::memory_order_relaxed) const {
    return detail::refcount(combined_refcount_.load(order));
  }

  uint32_t weakcount(
      std::memory_order order = std::memory_order_relaxed) const {
    return detail::weakcount(combined_refcount_.load(order));
  }
};

namespace detail {

#ifndef C10_MOBILE
template <>
struct TargetTraits<c10::intrusive_ptr_target> {
  static constexpr bool can_have_pyobject = true;
};
#endif

} // namespace detail

template <class TTarget, class NullType>
class weak_intrusive_ptr;

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      // NOLINTNEXTLINE(misc-redundant-expression)
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_base_of_v<
          TTarget,
          std::remove_pointer_t<decltype(NullType::singleton())>>,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;
  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  // 让 pybind11::class_ 成为 intrusive_ptr 的 friend class，这样 pybind11 中的 custom smart holder
  // 就能够访问 intrusive_ptr(T*) 的私有构造函数，该构造函数接管了对象的所有权。
  // 这是 custom holder macro PYBIND11_DECLARE_HOLDER_TYPE 所要求的，该 macro 使用
  // intrusive_ptr(TTarget*) 来初始化并接管对象的所有权。有关详细信息，请参见
  // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
  template <typename, typename...>
  friend class pybind11::class_;

  void retain_() noexcept {
    if (target_ != NullType::singleton()) {
      uint64_t combined = detail::atomic_combined_refcount_increment(
          target_->combined_refcount_, detail::kReferenceCountOne);
      uint32_t new_refcount = detail::refcount(combined);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");

      if constexpr (detail::TargetTraits<TTarget>::can_have_pyobject) {
        // 如果 refcount 从 1 转变为 2，我们需要 incref 该 PyObject。换句话说，
        // 我们需要确保 PyObject 保持存活，因为现在我们除了 PyObject 本身之外，
        // 还有一个指向此对象的 C++ reference。
        if (detail::has_pyobject(combined) && detail::refcount(combined) == 2) {
          target_->incref_pyobject();
        }
      } else {
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
            !detail::has_pyobject(combined),
            "TargetTraits indicates that type cannot have PyObject, but refcount has PyObject bit set.");
      }
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton()) {
      reset_not_null_(target_);
    }
  }

  // C10_NOINLINE 以保持二进制体积稍小。我们在这里传递 TTarget* 以避免在 reset_() 调用中额外的指针解引用。
  C10_NOINLINE static void reset_not_null_(TTarget* target) noexcept {
    if (detail::is_uniquely_owned(
            target->combined_refcount_.load(std::memory_order_acquire))) {
      // 两个计数都是 1，因此没有弱引用，并且我们正在释放最后一个强引用。
      // 没有其他线程能够无数据竞争地观察到该目标删除调用的效果（例如调用 use_count()）。
      target->combined_refcount_.store(0, std::memory_order_relaxed);
      delete target;
      return;
    }

    auto combined_refcount = detail::atomic_combined_refcount_decrement(
        target->combined_refcount_, detail::kReferenceCountOne);
    uint32_t new_refcount = detail::refcount(combined_refcount);
    bool has_pyobject = detail::has_pyobject(combined_refcount);
    if (new_refcount == 0) {
      if (detail::weakcount(combined_refcount) == 1) {
        delete target;
        return;
      }
      // 参见上面关于 weakcount 的注释。
      // 只要 refcount>0，weakcount 就比实际弱引用数大 1。因此我们需要在此递减它。
      release_resources_and_decrement_weakrefs_(target);
    } else if constexpr (detail::TargetTraits<TTarget>::can_have_pyobject) {
      // 如果 refcount 从 2 变为 1，我们需要减少 PyObject 的引用计数。
      // 换句话说，如果除了 PyObject 本身之外没有其他 C++ 引用指向此对象，
      // 我们不想让 PyObject 保持存活。
      if (has_pyobject && new_refcount == 1) {
        target->decref_pyobject();
      }
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          !has_pyobject,
          "TargetTraits indicates that type cannot have PyObject, but refcount has PyObject bit set.");
    }
  }

  C10_NOINLINE static void release_resources_and_decrement_weakrefs_(
      TTarget* target) noexcept {
    // const_cast 的理由：release_resources 基本上是一个析构函数，
    // 而析构函数总会修改对象，即使是 const 对象也是如此。
    const_cast<std::remove_const_t<TTarget>*>(target)->release_resources();
    if (detail::atomic_weakcount_decrement(target->combined_refcount_) == 0) {
      delete target;
    }
  }

  // 原始指针构造函数不是公开的，因为除了在 make_intrusive()、reclaim() 
  // 和 weak_intrusive_ptr::lock() 的实现内部之外，我们不应该通过原始指针构造 intrusive_ptr。

  // 此构造函数会为你增加引用计数。
  // 此构造函数将被 make_intrusive() 以及 pybind11 使用，
  // 后者将 intrusive_ptr 持有者包裹在原始指针周围
  // 并相应地进行 incref（pybind11 默认要求原始指针构造函数进行 incref）。
  explicit intrusive_ptr(TTarget* target)
      : intrusive_ptr(target, raw::DontIncreaseRefcount{}) {
    if (target_ != NullType::singleton()) {
      // 我们刚刚创建了 result.target_，因此我们知道没有其他线程可以访问它，
      // 所以我们知道不必关心内存顺序。
      // （在 x86_64 上，使用 memory_order_relaxed 的存储会生成普通的 `mov`，
      // 而原子增量会生成 lock 前缀的 `add`，后者昂贵得多：https://godbolt.org/z/eKPzj8。）
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          target_->combined_refcount_.load(std::memory_order_relaxed) == 0,
          "intrusive_ptr: Newly-created target had non-zero refcounts. Does its "
          "constructor do something strange like incref or create an "
          "intrusive_ptr from `this`?");
      target_->combined_refcount_.store(
          detail::kUniqueRef, std::memory_order_relaxed);
    }
  }

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  /* implicit */ intrusive_ptr(std::nullptr_t) noexcept
      : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

  // 此构造函数不会为你增加 ref counter。
  // 我们使用 tagged dispatch 机制来显式标记此构造函数不增加 refcount。
  explicit intrusive_ptr(
      TTarget* target,
      raw::DontIncreaseRefcount /*unused*/) noexcept
      : target_(target) {}

  explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
      : intrusive_ptr(rhs.release()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(const intrusive_ptr<From, FromNullType>& rhs)
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
    retain_();
  }

  ~intrusive_ptr() noexcept {
    reset_();
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    // NOLINTNEXTLINE(*assign*)
    return this->template operator= <TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
  intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  // 赋值通过拷贝和交换来实现
  // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
    // NOLINTNEXTLINE(*assign-operator, *assignment-signature)
    return this->template operator= <TTarget, NullType>(rhs);
  }

  template <class From, class FromNullType>
  intrusive_ptr& operator=(
      const intrusive_ptr<From, NullType>& rhs) & noexcept {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. intrusive_ptr copy assignment got pointer of wrong type.");
    intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  TTarget* get() const noexcept {
    return target_;
  }

  TTarget& operator*() const noexcept {
    return *target_;
  }

  TTarget* operator->() const noexcept {
    return target_;
  }

  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  void reset() noexcept {
    reset_();
    target_ = NullType::singleton();
  }

  void swap(intrusive_ptr& rhs) noexcept {
    std::swap(target_, rhs.target_);
  }

  // 我们在代码中做了很多空指针检查，让这个操作廉价是件好事。
  bool defined() const noexcept {
    return target_ != NullType::singleton();
  }

  uint32_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount(std::memory_order_relaxed);
  }

  uint32_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount(std::memory_order_relaxed);
  }

  // 只关心强引用是否为 1
  bool unique() const noexcept {
    return use_count() == 1;
  }

  /**
   * // 比 unique() 更强，因为它也必须没有任何弱引用。
   */
  bool is_uniquely_owned() const noexcept {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(target_ != NullType::singleton());
    return detail::is_uniquely_owned(
        target_->combined_refcount_.load(std::memory_order_acquire));
  }

  /**
   * 返回指向底层对象的所有权指针，并使 intrusive_ptr 实例失效。
   * 这意味着引用计数不会减少。
   * 你*必须*将返回的指针通过 intrusive_ptr::reclaim(ptr) 重新放回 intrusive_ptr 中以正确析构它。
   * 这对 C API 很有帮助。
   */
  TTarget* release() noexcept {
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * 接受一个指向 TTarget* 的所有权指针，并创建一个接管所有权的 intrusive_ptr。
   * 这意味着引用计数不会增加。
   * 这是 intrusive_ptr::release() 的对应方法，传入的指针*必须*是通过 intrusive_ptr::release() 创建的。
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_ptr == NullType::singleton() || owning_ptr->refcount() == 0 ||
            owning_ptr->weakcount(),
        "TTarget violates the invariant that refcount > 0  =>  weakcount > 0");
    return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{});
  }

  /**
   * 接受一个指向 TTarget* 的所有权指针，并创建一个表示新引用的 intrusive_ptr，
   * 即原始指针保留所有权。
   */
  static intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  /**
   * 在堆上分配一个对象，使用参数构造它，并将其包装在 intrusive_ptr 中并增加引用计数。
   * 这是一个辅助函数，用于让 make_intrusive() 访问私有的 intrusive_ptr 构造函数。
   */
  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    return intrusive_ptr(new TTarget(std::forward<Args>(args)...));
  }

  /**
   * 将一个 TTarget 的新实例（例如，使用 new TTarget(...) 分配的对象）转换为 intrusive_ptr。
   * 如果可能，请使用 intrusive_ptr::make，它能在静态层面保证分配正确完成。
   *
   * 目前，此方法存在的唯一原因是 pybind11 的持有者类型期望能够以这种方式进行分配（因为 pybind11 自己处理 new 分配）。
   */
  static intrusive_ptr unsafe_steal_from_new(TTarget* raw_ptr) {
    return intrusive_ptr(raw_ptr);
  }

  /**
   * 将一个不应进行引用计数（例如，通过 placement new 在 arena 中分配）的 TTarget 实例转换为 intrusive_ptr。
   * 这是完全不安全的，仅在你能够保证该指针不会逃逸并被正常引用计数时才能使用。
   *
   * `expected_decrefs` 是一个调试参数：它指示所涉及的 intrusive_ptr_target 预期的强所有者数量。
   * 在大多数用例中，这很可能是 1。
   *
   * 此方法存在的理由是在静态运行时中手动跨 Tensor 共享 StorageImpl。
   * 它需要访问私有的 intrusive_ptr 成员，以便将引用计数初始化为自定义值。
   */
  static intrusive_ptr unsafe_adapt_non_heap_allocated(
      TTarget* raw_ptr,
      uint32_t expected_decrefs) {
    intrusive_ptr result(raw_ptr, raw::DontIncreaseRefcount{});
    // kImpracticallyHugeReferenceCount 对于引用计数来说是不切实际的大，但同时不会溢出 uint32_t。
    // 我们实际上只需要将引用计数初始化为 2 —— 我们只是进行一个不平衡的 incref 以防止非堆分配的目标被释放，
    // 并且我们通过直接初始化引用计数而不是进行昂贵的原子增量来优化这个 incref。
    // 使用 kImpracticallyHugeReferenceCount 的原因是为了适应 ~intrusive_ptr_target 中的调试断言。
#ifdef NDEBUG
    expected_decrefs = 0;
#endif
    result.target_->combined_refcount_.store(
        detail::refcount(
            detail::kImpracticallyHugeReferenceCount + expected_decrefs) |
            detail::kImpracticallyHugeWeakReferenceCount,
        std::memory_order_relaxed);
    return result;
  }

  /**
   * 将一个**非所有权的原始指针**转换为 intrusive_ptr。
   * 它在道德上等同于 shared_ptr 上的 enable_shared_from_this。
   *
   * 此方法仅对已经存活的对象有效。
   * 如果你在寻找 unique_ptr<T>(T*) 构造函数的道德等价物，请参见 steal_from_new。
   *
   * TODO: https://github.com/pytorch/pytorch/issues/56482
   */
  static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        raw_ptr == NullType::singleton() || raw_ptr->refcount() > 0,
        "intrusive_ptr: Can only reclaim pointers that are owned by someone");
    auto ptr = reclaim(raw_ptr); // doesn't increase refcount
    ptr.retain_();
    return ptr;
  }
};

// 构建 intrusive_ptr
template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

// 交换两个 intrusive_ptr
template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// 为了允许 intrusive_ptr 在 std::map 或 std::set 里，我们需要重载运算符 < 和 =
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <class TTarget1, class NullType1>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return lhs.get() == nullptr;
}

template <class TTarget2, class NullType2>
inline bool operator==(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return nullptr == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

template <class TTarget1, class NullType1>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    std::nullptr_t) noexcept {
  return !operator==(lhs, nullptr);
}

template <class TTarget2, class NullType2>
inline bool operator!=(
    std::nullptr_t,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(nullptr, rhs);
}

// 定义零成本借用语义
template <typename T>
struct MaybeOwnedTraits<c10::intrusive_ptr<T>> {
  using owned_type = c10::intrusive_ptr<T>;
  using borrow_type = c10::intrusive_ptr<T>;

  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type::reclaim(from.get());
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.release();
    lhs = borrow_type::reclaim(rhs.get());
  }

  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.release();
  }

  static const owned_type& referenceFromBorrow(
      const borrow_type& borrow) noexcept {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(
      const borrow_type& borrow) noexcept {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) noexcept {
    return true;
  }
};

template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of_v<intrusive_ptr_target, TTarget>,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
#ifndef _WIN32
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_base_of_v<
          TTarget,
          std::remove_pointer_t<decltype(NullType::singleton())>>,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;

  void retain_() {
    if (target_ != NullType::singleton()) {
      uint32_t new_weakcount =
          detail::atomic_weakcount_increment(target_->combined_refcount_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_weakcount != 1,
          "weak_intrusive_ptr: Cannot increase weakcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() &&
        detail::atomic_weakcount_decrement(target_->combined_refcount_) == 0) {
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
      delete target_;
    }
    target_ = NullType::singleton();
  }

  constexpr explicit weak_intrusive_ptr(TTarget* target) : target_(target) {}

 public:
  using element_type = TTarget;

  explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& ptr)
      : weak_intrusive_ptr(ptr.get()) {
    retain_();
  }

  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
      weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      const weak_intrusive_ptr<From, FromNullType>& rhs)
      : target_(
            detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
    retain_();
  }

  ~weak_intrusive_ptr() noexcept {
    reset_();
  }

  weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept {
    // NOLINTNEXTLINE(*assign*)
    return this->template operator= <TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
  weak_intrusive_ptr& operator=(
      weak_intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. weak_intrusive_ptr move assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
    if (this == &rhs) {
      return *this;
    }
    // NOLINTNEXTLINE(*assign*)
    return this->template operator= <TTarget, NullType>(rhs);
  }

  weak_intrusive_ptr& operator=(
      const intrusive_ptr<TTarget, NullType>& rhs) & noexcept {
    weak_intrusive_ptr tmp(rhs);
    swap(tmp);
    return *this;
  }

  template <class From, class FromNullType>
  weak_intrusive_ptr& operator=(
      const weak_intrusive_ptr<From, NullType>& rhs) & noexcept {
    static_assert(
        std::is_convertible_v<From*, TTarget*>,
        "Type mismatch. weak_intrusive_ptr copy assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  void reset() noexcept {
    reset_();
  }

  void swap(weak_intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // 注意：这应该 ONLY 被 weak_intrusive_ptr 的 std::hash 实现使用。
  // 另一种做法是将 friend std::hash<weak_intrusive_ptr> 声明为友元，但这会触发两个 bug：
  //
  //  (1) 它会触发一个 nvcc 的 bug，在友元类声明中的 std::hash 会被预处理为 hash，
  //      然后实际上无法找到。这个错误的报错信息类似于：
  //
  //        error: no template named 'hash'; did you mean 'std::hash'?
  //
  //  (2) 在 OS X 上，std::hash 被声明为 struct，而不是 class。这会触发：
  //
  //        error: class 'hash' was previously declared as a struct
  //        [-Werror,-Wmismatched-tags]
  //
  // 这两个问题都可以被解决，但总的来说，我觉得如果我们只是暴露一个
  // 用于获取 target_ 的不安全 getter，会更简单、更容易使其工作。
  TTarget* _unsafe_get_target() const noexcept {
    return target_;
  }

  uint32_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount(
        std::memory_order_relaxed); // refcount, not weakcount!
  }

  uint32_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount(std::memory_order_relaxed);
  }

  bool expired() const noexcept {
    return use_count() == 0;
  }

  // 从 weak_intrusive_ptr 安全地升级为 intrusive_ptr
  intrusive_ptr<TTarget, NullType> lock() const noexcept {
    if (target_ == NullType::singleton()) {
      return intrusive_ptr<TTarget, NullType>();
    } else {
      bool increfed = false;
      auto combined_refcount =
          target_->combined_refcount_.load(std::memory_order_relaxed);
      do {
        if (detail::refcount(combined_refcount) == 0) {
          // 对象已经被销毁，返回 nullptr
          return intrusive_ptr<TTarget, NullType>();
        }
        if constexpr (detail::TargetTraits<TTarget>::can_have_pyobject) {
          if (detail::has_pyobject(combined_refcount) &&
              detail::refcount(combined_refcount) == 1 && !increfed) {
            // 对象有一个 Python 包装器，并且没有其他 C++ 引用。
            // 我们需要在获取 C++ 对象的强引用之前增加 Python 对象的引用计数，
            // 以避免出现 Python 对象被并发释放的情况。
            if (!target_->try_incref_pyobject()) {
              return intrusive_ptr<TTarget, NullType>();
            }
            increfed = true;
          }
        }
        // 成功时计数+1，必须确保能看到之前其他线程在释放引用计数（release）
        // 失败时重试，不涉及同步数据，直接 relaxed
      } while (!target_->combined_refcount_.compare_exchange_weak(
          combined_refcount,
          combined_refcount + detail::kReferenceCountOne,
          std::memory_order_acquire,
          std::memory_order_relaxed));

      // 由于前面多加了一次 python 计数，所以这里要减回来
      if constexpr (detail::TargetTraits<TTarget>::can_have_pyobject) {
        if (increfed && detail::refcount(combined_refcount) != 1) {
          target_->decref_pyobject();
        }
      }

      return intrusive_ptr<TTarget, NullType>(
          target_, raw::DontIncreaseRefcount{});
    }
  }

  /**
   * 返回指向底层对象的所有权（但仍然是弱引用）指针，并使 weak_intrusive_ptr 实例失效。
   * 这意味着弱引用计数不会减少。
   * 你*必须*将返回的指针通过 weak_intrusive_ptr::reclaim(ptr) 重新放回 weak_intrusive_ptr 中以正确析构它。
   * 这对 C API 很有帮助。
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * 接受一个所有权（但必须是弱引用的）指向 TTarget* 的指针，并创建一个接管所有权的 weak_intrusive_ptr。
   * 这意味着弱引用计数不会增加。
   * 这是 weak_intrusive_ptr::release() 的对应方法，传入的指针*必须*是通过 weak_intrusive_ptr::release() 创建的。
   */
  static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
    // 参见注释 [栈上分配的 intrusive_ptr_target 安全性]
    // 如果 refcount > 0，则 weakcount 必须 >1 才能存在弱引用。
    // 参见本文件开头的弱计数解释。
    // 如果 refcount == 0，则 weakcount 必须 >0。
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_weak_ptr == NullType::singleton() ||
            owning_weak_ptr->weakcount() > 1 ||
            (owning_weak_ptr->refcount() == 0 &&
             owning_weak_ptr->weakcount() > 0),
        "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release().");
    return weak_intrusive_ptr(owning_weak_ptr);
  }

  /**
   * 接受一个指向 TTarget* 的指针（可能是弱引用或强引用），并创建一个表示新弱引用的新 weak_intrusive_ptr，
   * 即原始指针保留所有权。
   */
  static weak_intrusive_ptr reclaim_copy(TTarget* owning_ptr) {
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator<(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator==(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
};

template <class TTarget, class NullType>
inline void swap(
    weak_intrusive_ptr<TTarget, NullType>& lhs,
    weak_intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// To allow weak_intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ < rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ == rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

// 为了文档目的而起的别名，以便更容易区分弱原始 intrusive 指针和 intrusive 指针。
using weak_intrusive_ptr_target = intrusive_ptr_target;

// 此命名空间提供了一些方法来处理继承自 intrusive_ptr_target 的原始指针
// 正常不会用到，不必理会
namespace raw {

namespace intrusive_ptr {

inline void incref(intrusive_ptr_target* self) {
  if (self) {
    uint64_t combined = detail::atomic_combined_refcount_increment(
        self->combined_refcount_, detail::kReferenceCountOne);

#ifndef C10_MOBILE
    if (detail::has_pyobject(combined) && detail::refcount(combined) == 2) {
      self->incref_pyobject();
    }
#else
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!detail::has_pyobject(combined));
#endif
  }
}

inline void decref(intrusive_ptr_target* self) {
  c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
}

template <typename T>
inline T* make_weak(T* self) {
  auto ptr = c10::intrusive_ptr<T>::reclaim(self);
  c10::weak_intrusive_ptr<T> wptr(ptr);
  ptr.release();
  return wptr.release();
}

inline uint32_t use_count(intrusive_ptr_target* self) {
  auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  auto r = ptr.use_count();
  ptr.release();
  return r;
}

} // namespace intrusive_ptr

namespace weak_intrusive_ptr {

inline void incref(weak_intrusive_ptr_target* self) {
  detail::atomic_weakcount_increment(self->combined_refcount_);
}

inline void decref(weak_intrusive_ptr_target* self) {
  c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
}

template <typename T>
inline T* lock(T* self) {
  auto wptr = c10::weak_intrusive_ptr<T>::reclaim(self);
  auto ptr = wptr.lock();
  wptr.release();
  return ptr.release();
}

inline uint32_t use_count(weak_intrusive_ptr_target* self) {
  auto wptr = c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
  auto r = wptr.use_count();
  wptr.release();
  return r;
}

} // namespace weak_intrusive_ptr

} // namespace raw

} // namespace c10

namespace std {
// 为了在 std::unordered_map 或 std::unordered_set 中使用 intrusive_ptr 和 weak_intrusive_ptr，我们需要 std::hash
template <class TTarget, class NullType>
struct hash<c10::intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
template <class TTarget, class NullType>
struct hash<c10::weak_intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::weak_intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x._unsafe_get_target());
  }
};
} // namespace std
