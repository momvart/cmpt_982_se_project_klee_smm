//===-- Memory.cpp --------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"

#include "Context.h"
#include "ExecutionState.h"
#include "MemoryManager.h"

#include "klee/ADT/BitArray.h"
#include "klee/Expr/ArrayCache.h"
#include "klee/Expr/Expr.h"
#include "klee/Support/OptionCategories.h"
#include "klee/Solver/Solver.h"
#include "klee/Support/ErrorHandling.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <sstream>
#include <algorithm>

using namespace llvm;
using namespace klee;

namespace {
  cl::opt<bool>
  UseConstantArrays("use-constant-arrays",
                    cl::desc("Use constant arrays instead of updates when possible (default=true)\n"),
                    cl::init(true),
                    cl::cat(SolvingCat));
}

/***/

int MemoryObject::counter = 0;

MemoryObject::~MemoryObject() {
  if (parent)
    parent->markFreed(this);
}

void MemoryObject::getAllocInfo(std::string &result) const {
  llvm::raw_string_ostream info(result);

  info << "MO" << id << "[" << size << "]";

  if (allocSite) {
    info << " allocated at ";
    if (const Instruction *i = dyn_cast<Instruction>(allocSite)) {
      info << i->getParent()->getParent()->getName() << "():";
      info << *i;
    } else if (const GlobalValue *gv = dyn_cast<GlobalValue>(allocSite)) {
      info << "global:" << gv->getName();
    } else {
      info << "value:" << *allocSite;
    }
  } else {
    info << " (no allocation info)";
  }
  
  info.flush();
}

/***/

ObjectState::ObjectState(const MemoryObject *mo)
  : copyOnWriteOwner(0),
    refCount(0),
    object(mo),
    concreteStore(new uint8_t[mo->size]),
    concreteMask(nullptr),
    knownSymbolics(nullptr),
    unflushedMask(nullptr),
    updates(nullptr, nullptr),
    size(mo->size),
    readOnly(false) {
  if (!UseConstantArrays) {
    static unsigned id = 0;
    const Array *array =
        getArrayCache()->CreateArray("tmp_arr" + llvm::utostr(++id), size);
    updates = UpdateList(array, 0);
  }
  memset(concreteStore, 0, size);
}


ObjectState::ObjectState(const MemoryObject *mo, const Array *array)
  : copyOnWriteOwner(0),
    refCount(0),
    object(mo),
    concreteStore(new uint8_t[mo->size]),
    concreteMask(nullptr),
    knownSymbolics(nullptr),
    unflushedMask(nullptr),
    updates(array, nullptr),
    size(mo->size),
    readOnly(false) {
  makeSymbolic();
  memset(concreteStore, 0, size);
}

ObjectState::ObjectState(const ObjectState &os) 
  : copyOnWriteOwner(0),
    refCount(0),
    concreteStore(new uint8_t[os.size]),
    concreteMask(os.concreteMask ? new BitArray(*os.concreteMask, os.size) : 0),
    flushMask(os.flushMask ? new BitArray(*os.flushMask, os.size) : 0),
    knownSymbolics(0),
    updates(os.updates),
    object(os.object),
    size(os.size),
    readOnly(false) {
  assert(!os.readOnly && "no need to copy read only object?");
  if(size != object->size) fprintf(stderr, "os size %u, mo size %u\n", size, object->size);
  assert(size == object->size && "Object state size doesn't match memory object size");
  if (object)
    object->refCount++;

  if (os.knownSymbolics) {
    knownSymbolics = new ref<Expr>[size];
    for (unsigned i=0; i<size; i++)
      knownSymbolics[i] = os.knownSymbolics[i];
  }

  memcpy(concreteStore, os.concreteStore, size*sizeof(*concreteStore));
}

ObjectState::ObjectState(const ObjectState &os, const MemoryObject* mo) 
  : copyOnWriteOwner(0),
    refCount(0),
    concreteStore(new uint8_t[os.size]),
    concreteMask(os.concreteMask ? new BitArray(*os.concreteMask, os.size) : nullptr),
    knownSymbolics(nullptr),
    unflushedMask(os.unflushedMask ? new BitArray(*os.unflushedMask, os.size) : nullptr),
    updates(os.updates),
    object(mo),
    size(os.size),
    readOnly(false) {
  assert(!os.readOnly && "no need to copy read only object?");
  if(size != object->size) fprintf(stderr, "os size %u, mo size %u\n", size, object->size);
  assert(size == object->size && "Object state size doesn't match memory object size");
  if (object)
    object->refCount++;

  if (os.knownSymbolics) {
    knownSymbolics = new ref<Expr>[size];
    for (unsigned i=0; i<size; i++)
      knownSymbolics[i] = os.knownSymbolics[i];
  }

  memcpy(concreteStore, os.concreteStore, size*sizeof(*concreteStore));
}

ObjectState::~ObjectState() {
  delete concreteMask;
  delete unflushedMask;
  delete[] knownSymbolics;
  delete[] concreteStore;
}

ArrayCache *ObjectState::getArrayCache() const {
  assert(object && "object was NULL");
  return object->parent->getArrayCache();
}

/***/
void ObjectState::realloc(unsigned int newSize) {
         if(newSize == size) return;
          uint8_t *store = new uint8_t[newSize];
          memcpy(store, concreteStore, std::min(size,newSize));
//          printf("Os %p Realloc size %d, newSize %d from %p to %p, concreteMaskL %p flushMask %p\n", this, size, newSize, concreteStore, store, concreteMask, flushMask);
//leads to double free for some reason
//          delete[] concreteStore;
          if(concreteMask != nullptr) {
              BitArray *cm;
              //printf("Updating concrete mask old size: %d, new size %d, bits %p\n", size, newSize, concreteMask);
              cm = new BitArray(*concreteMask, newSize,std::min(size,newSize));
              if(newSize > size) {
                  for(unsigned i = size; i < newSize; i++) {
                      cm->set(i);
                  }
              }
              delete concreteMask;
              concreteMask = cm;
          }
          if(flushMask != nullptr) {
              BitArray *fm;
              //printf("Updating flush mask %p bits: %p\n", flushMask, flushMask->bits);
              fm = new BitArray(*flushMask, newSize,std::min(size,newSize));
              delete flushMask;
              flushMask = fm;
          }
          if(knownSymbolics != nullptr) {
            ref<Expr> *kS;
            //printf("Updating known symbolics %p\n", knownSymbolics);
            kS = new ref<Expr>[newSize];
            for(int i = 0; i < std::min(size,newSize); i++) {
                kS[i] = knownSymbolics[i];
            }
            delete[] knownSymbolics;
            knownSymbolics = kS;

          }
          if(updates.root != nullptr) {
             //printf("Updates.root size: %u, name %s\n", updates.root->size, updates.root->name.c_str());
             const Array* newRoot = getArrayCache()->CreateResizedArray(updates.root, newSize);
             updates = UpdateList(newRoot, updates.head);
          }
          delete[] concreteStore;
          concreteStore = store;
          size = newSize;


}

const UpdateList &ObjectState::getUpdates() const {
  // Constant arrays are created lazily.
  if (!updates.root) {
    // Collect the list of writes, with the oldest writes first.
    
    // FIXME: We should be able to do this more efficiently, we just need to be
    // careful to get the interaction with the cache right. In particular we
    // should avoid creating UpdateNode instances we never use.
    unsigned NumWrites = updates.head ? updates.head->getSize() : 0;
    std::vector< std::pair< ref<Expr>, ref<Expr> > > Writes(NumWrites);
    const auto *un = updates.head.get();
    for (unsigned i = NumWrites; i != 0; un = un->next.get()) {
      --i;
      Writes[i] = std::make_pair(un->index, un->value);
    }

    std::vector< ref<ConstantExpr> > Contents(size);

    // Initialize to zeros.
    for (unsigned i = 0, e = size; i != e; ++i)
      Contents[i] = ConstantExpr::create(0, Expr::Int8);

    // Pull off as many concrete writes as we can.
    std::vector<unsigned> skippedIndices;
    unsigned Begin = 0, End = Writes.size();
    for (; Begin != End; ++Begin) {
      // Push concrete writes into the constant array.
      ConstantExpr *Index = dyn_cast<ConstantExpr>(Writes[Begin].first);
      if (!Index)
        break;

      ConstantExpr *Value = dyn_cast<ConstantExpr>(Writes[Begin].second);
      if (!Value) {
        skippedIndices.push_back(Begin);
        continue;
      }

      Contents[Index->getZExtValue()] = Value;
    }

    static unsigned id = 0;
    const Array *array = getArrayCache()->CreateArray(
        "const_arr" + llvm::utostr(++id), size, &Contents[0],
        &Contents[0] + Contents.size());
    updates = UpdateList(array, 0);

    for(unsigned B : skippedIndices) {
      updates.extend(Writes[B].first, Writes[B].second);
    }

    // Apply the remaining (non-constant) writes.
    for (; Begin != End; ++Begin)
      updates.extend(Writes[Begin].first, Writes[Begin].second);
  }

  return updates;
}

void ObjectState::flushToConcreteStore(TimingSolver *solver,
                                       const ExecutionState &state) const {
  for (unsigned i = 0; i < size; i++) {
    if (isByteKnownSymbolic(i)) {
      ref<ConstantExpr> ce;
      bool success = solver->getValue(state.constraints, read8(i), ce,
                                      state.queryMetaData);
      if (!success)
        klee_warning("Solver timed out when getting a value for external call, "
                     "byte %p+%u will have random value",
                     (void *)object->address, i);
      else
        ce->toMemory(concreteStore + i);
    }
  }
}

void ObjectState::makeConcrete() {
  delete concreteMask;
  delete unflushedMask;
  delete[] knownSymbolics;
  concreteMask = nullptr;
  unflushedMask = nullptr;
  knownSymbolics = nullptr;
}

void ObjectState::makeSymbolic() {
  assert(!updates.head &&
         "XXX makeSymbolic of objects with symbolic values is unsupported");

  // XXX simplify this, can just delete various arrays I guess
  for (unsigned i=0; i<size; i++) {
    markByteSymbolic(i);
    setKnownSymbolic(i, 0);
    markByteFlushed(i);
  }
}


void ObjectState::initializeToZero() {
  makeConcrete();
  memset(concreteStore, 0, size);
}

void ObjectState::initializeToRandom() {  
  makeConcrete();
  for (unsigned i=0; i<size; i++) {
    // randomly selected by 256 sided die
    concreteStore[i] = 0xAB;
  }
}

/*
Cache Invariants
--
isByteKnownSymbolic(i) => !isByteConcrete(i)
isByteConcrete(i) => !isByteKnownSymbolic(i)
isByteUnflushed(i) => (isByteConcrete(i) || isByteKnownSymbolic(i))
 */

void ObjectState::fastRangeCheckOffset(ref<Expr> offset,
                                       unsigned *base_r,
                                       unsigned *size_r) const {
  *base_r = 0;
  *size_r = size;
}

void ObjectState::flushRangeForRead(unsigned rangeBase,
                                    unsigned rangeSize) const {
  if (!unflushedMask)
    unflushedMask = new BitArray(size, true);

  for (unsigned offset = rangeBase; offset < rangeBase + rangeSize; offset++) {
    if (isByteUnflushed(offset)) {
      if (isByteConcrete(offset)) {
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       ConstantExpr::create(concreteStore[offset], Expr::Int8));
      } else {
        assert(isByteKnownSymbolic(offset) &&
               "invalid bit set in unflushedMask");
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       knownSymbolics[offset]);
      }

      unflushedMask->unset(offset);
    }
  }
}

void ObjectState::flushRangeForWrite(unsigned rangeBase, unsigned rangeSize) {
  if (!unflushedMask)
    unflushedMask = new BitArray(size, true);

  for (unsigned offset = rangeBase; offset < rangeBase + rangeSize; offset++) {
    if (isByteUnflushed(offset)) {
      if (isByteConcrete(offset)) {
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       ConstantExpr::create(concreteStore[offset], Expr::Int8));
        markByteSymbolic(offset);
      } else {
        assert(isByteKnownSymbolic(offset) &&
               "invalid bit set in unflushedMask");
        updates.extend(ConstantExpr::create(offset, Expr::Int32),
                       knownSymbolics[offset]);
        setKnownSymbolic(offset, 0);
      }

      unflushedMask->unset(offset);
    } else {
      // flushed bytes that are written over still need
      // to be marked out
      if (isByteConcrete(offset)) {
        markByteSymbolic(offset);
      } else if (isByteKnownSymbolic(offset)) {
        setKnownSymbolic(offset, 0);
      }
    }
  }
}

bool ObjectState::isByteConcrete(unsigned offset) const {
  return !concreteMask || concreteMask->get(offset);
}

bool ObjectState::isByteUnflushed(unsigned offset) const {
  return !unflushedMask || unflushedMask->get(offset);
}

bool ObjectState::isByteKnownSymbolic(unsigned offset) const {
  return knownSymbolics && knownSymbolics[offset].get();
}

void ObjectState::markByteConcrete(unsigned offset) {
  if (concreteMask)
    concreteMask->set(offset);
}

void ObjectState::markByteSymbolic(unsigned offset) {
  if (!concreteMask) {
    concreteMask = new BitArray(size, true);
  }
  concreteMask->unset(offset);
}

void ObjectState::markByteUnflushed(unsigned offset) {
  if (unflushedMask)
    unflushedMask->set(offset);
}

void ObjectState::markByteFlushed(unsigned offset) {
  if (!unflushedMask) {
    unflushedMask = new BitArray(size, false);
  } else {
    unflushedMask->unset(offset);
  }
}

void ObjectState::setKnownSymbolic(unsigned offset, 
                                   Expr *value /* can be null */) {
  if (knownSymbolics) {
    knownSymbolics[offset] = value;
  } else {
    if (value) {
      knownSymbolics = new ref<Expr>[size];
      knownSymbolics[offset] = value;
    }
  }
}

/***/

ref<Expr> ObjectState::read8(unsigned offset) const {
  if (isByteConcrete(offset)) {
    return ConstantExpr::create(concreteStore[offset], Expr::Int8);
  } else if (isByteKnownSymbolic(offset)) {
    return knownSymbolics[offset];
  } else {
    assert(!isByteUnflushed(offset) && "unflushed byte without cache value");
    
    return ReadExpr::create(getUpdates(), 
                            ConstantExpr::create(offset, Expr::Int32));
  }    
}

ref<Expr> ObjectState::read8(ref<Expr> offset) const {
  assert(!isa<ConstantExpr>(offset) &&
         "constant offset passed to symbolic read8");
  unsigned base, size;
  fastRangeCheckOffset(offset, &base, &size);
  flushRangeForRead(base, size);

  if (size > 4096) {
    std::string allocInfo;
    object->getAllocInfo(allocInfo);
    klee_warning_once(
        nullptr,
        "Symbolic memory access will send the following array of %d bytes to "
        "the constraint solver -- large symbolic arrays may cause significant "
        "performance issues: %s",
        size, allocInfo.c_str());
  }

  return ReadExpr::create(getUpdates(), ZExtExpr::create(offset, Expr::Int32));
}

void ObjectState::write8(unsigned offset, uint8_t value) {
  //assert(read_only == false && "writing to read-only object!");
  concreteStore[offset] = value;
  setKnownSymbolic(offset, 0);

  markByteConcrete(offset);
  markByteUnflushed(offset);
}

void ObjectState::write8(unsigned offset, ref<Expr> value) {
  // can happen when ExtractExpr special cases
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    write8(offset, (uint8_t) CE->getZExtValue(8));
  } else {
    unsigned i = offset;
    assert(value.get() != NULL && "NULL value breaks invariants");
    setKnownSymbolic(offset, value.get());
      
    markByteSymbolic(offset);
    markByteUnflushed(offset);
  }
}

void ObjectState::write8(ref<Expr> offset, ref<Expr> value) {
  assert(!isa<ConstantExpr>(offset) &&
         "constant offset passed to symbolic write8");
  unsigned base, size;
  fastRangeCheckOffset(offset, &base, &size);
  flushRangeForWrite(base, size);

  if (size > 4096) {
    std::string allocInfo;
    object->getAllocInfo(allocInfo);
    klee_warning_once(
        nullptr,
        "Symbolic memory access will send the following array of %d bytes to "
        "the constraint solver -- large symbolic arrays may cause significant "
        "performance issues: %s",
        size, allocInfo.c_str());
  }

  updates.extend(ZExtExpr::create(offset, Expr::Int32), value);
}

/***/

ref<Expr> ObjectState::read(ref<Expr> offset, Expr::Width width) const {
  // Truncate offset to 32-bits.
  offset = ZExtExpr::create(offset, Expr::Int32);

  // Check for reads at constant offsets.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(offset))
    return read(CE->getZExtValue(32), width);

  // Treat bool specially, it is the only non-byte sized write we allow.
  if (width == Expr::Bool)
    return ExtractExpr::create(read8(offset), 0, Expr::Bool);

  // Otherwise, follow the slow general case.
  unsigned NumBytes = width / 8;
  assert(width == NumBytes * 8 && "Invalid read size!");
  ref<Expr> Res(0);
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    ref<Expr> Byte = read8(AddExpr::create(offset, 
                                           ConstantExpr::create(idx, 
                                                                Expr::Int32)));
    Res = i ? ConcatExpr::create(Byte, Res) : Byte;
  }

  return Res;
}

ref<Expr> ObjectState::read(unsigned offset, Expr::Width width) const {
  // Treat bool specially, it is the only non-byte sized write we allow.
  if (width == Expr::Bool)
    return ExtractExpr::create(read8(offset), 0, Expr::Bool);

  // Otherwise, follow the slow general case.
  unsigned NumBytes = width / 8;
  assert(width == NumBytes * 8 && "Invalid width for read size!");
  ref<Expr> Res(0);
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    ref<Expr> Byte = read8(offset + idx);
    Res = i ? ConcatExpr::create(Byte, Res) : Byte;
  }

  return Res;
}

void ObjectState::write(ref<Expr> offset, ref<Expr> value) {
  // Truncate offset to 32-bits.
  offset = ZExtExpr::create(offset, Expr::Int32);

  // Check for writes at constant offsets.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(offset)) {
    write(CE->getZExtValue(32), value);
    return;
  }

  // Treat bool specially, it is the only non-byte sized write we allow.
  Expr::Width w = value->getWidth();
  if (w == Expr::Bool) {
    write8(offset, ZExtExpr::create(value, Expr::Int8));
    return;
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = w / 8;
  assert(w == NumBytes * 8 && "Invalid write size!");
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(AddExpr::create(offset, ConstantExpr::create(idx, Expr::Int32)),
           ExtractExpr::create(value, 8 * i, Expr::Int8));
  }
}

void ObjectState::write(unsigned offset, ref<Expr> value) {
  // Check for writes of constant values.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(value)) {
    Expr::Width w = CE->getWidth();
    if (w <= 64 && klee::bits64::isPowerOfTwo(w)) {
      uint64_t val = CE->getZExtValue();
      switch (w) {
      default: assert(0 && "Invalid write size!");
      case  Expr::Bool:
      case  Expr::Int8:  write8(offset, val); return;
      case Expr::Int16: write16(offset, val); return;
      case Expr::Int32: write32(offset, val); return;
      case Expr::Int64: write64(offset, val); return;
      }
    }
  }

  // Treat bool specially, it is the only non-byte sized write we allow.
  Expr::Width w = value->getWidth();
  if (w == Expr::Bool) {
    write8(offset, ZExtExpr::create(value, Expr::Int8));
    return;
  }

  // Otherwise, follow the slow general case.
  unsigned NumBytes = w / 8;
  assert(w == NumBytes * 8 && "Invalid write size!");
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, ExtractExpr::create(value, 8 * i, Expr::Int8));
  }
} 

void ObjectState::write16(unsigned offset, uint16_t value) {
  unsigned NumBytes = 2;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::write32(unsigned offset, uint32_t value) {
  unsigned NumBytes = 4;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::write64(unsigned offset, uint64_t value) {
  unsigned NumBytes = 8;
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned idx = Context::get().isLittleEndian() ? i : (NumBytes - i - 1);
    write8(offset + idx, (uint8_t) (value >> (8 * i)));
  }
}

void ObjectState::print() const {
  llvm::errs() << "-- ObjectState --\n";
  llvm::errs() << "\tMemoryObject ID: " << object->id << "\n";
  llvm::errs() << "\tRoot Object: " << updates.root << "\n";
  llvm::errs() << "\tSize: " << size << "\n";

  llvm::errs() << "\tBytes:\n";
  for (unsigned i=0; i<size; i++) {
    llvm::errs() << "\t\t["<<i<<"]"
               << " concrete? " << isByteConcrete(i)
               << " known-sym? " << isByteKnownSymbolic(i)
               << " unflushed? " << isByteUnflushed(i) << " = ";
    ref<Expr> e = read8(i);
    llvm::errs() << e << "\n";
  }

  llvm::errs() << "\tUpdates:\n";
  for (const auto *un = updates.head.get(); un; un = un->next.get()) {
    llvm::errs() << "\t\t[" << un->index << "] = " << un->value << "\n";
  }
}


void FreeOffsets::addFreeSpace(unsigned offset, unsigned size) {
    auto p = std::make_pair(offset, size);
    freeObjects.insert(p);
    auto firstNextOffset = freeObjects.upper_bound(p);
    if(firstNextOffset != freeObjects.end()) {
//        fprintf(stderr,"next offset %u, offset + size %u\n", firstNextOffset->first, offset + size);
        if(firstNextOffset->first == offset + size) {
//            fprintf(stderr,"Found adjcent at %u with size %u to %u\n", offset, size, firstNextOffset->first);
            freeObjects.erase(freeObjects.find(p));
            p.second += firstNextOffset->second;
            freeObjects.insert(p);
            freeObjects.erase(firstNextOffset);
        }
    }
}
#define MAX_SIZE 20000000
int FreeOffsets::findFreeSpace(unsigned size) { 
    unsigned currentLowestSize = MAX_SIZE;
    std::pair<unsigned,unsigned> lowestSpace;
    for(auto it = freeObjects.begin(); it != freeObjects.end(); it++) {
        unsigned cSize = it->second;
        unsigned cOffset = it->first;

        if(cSize == size) { //perfect fit we are done
          freeObjects.erase(it);
          return cOffset;
        } else if (cSize > size && cSize < currentLowestSize) { //remember and keep looking
          lowestSpace = *it;
          currentLowestSize = cSize;
        }
    }

    if(currentLowestSize != MAX_SIZE) { // we found something
        unsigned ret = lowestSpace.first;
        freeObjects.erase(freeObjects.find(lowestSpace));
        
        lowestSpace.first += size;
        lowestSpace.second -= size;
        freeObjects.insert(lowestSpace);
        return ret;
    }
    return -1;

}
unsigned FreeOffsets::totalFreeSpace() {
    unsigned totalSize = 0;
    unsigned count = 0;
    std::vector<unsigned> sizes(freeObjects.size());
    for(auto& offsetSize : freeObjects) {
        totalSize += offsetSize.second;
        sizes[count] = offsetSize.second;
        count++;
    }
    std::sort(sizes.begin(), sizes.end());
    fprintf(stderr, "total size: %u, count %u, mean %f median: %u\n", totalSize, count, totalSize / (double)count, sizes[count / 2]);
    return totalSize;
}
