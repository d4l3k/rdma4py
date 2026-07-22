#include <stdint.h>

#define DOCA_GPUNETIO_VERBS_MKEY_SWAPPED 0
#include <doca_gpunetio_dev_verbs_qp.cuh>

namespace {

constexpr auto kSharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
constexpr auto kHandler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;

__device__ __forceinline__ doca_gpu_dev_verbs_qp *qp_from_u64(uint64_t qp)
{
	return reinterpret_cast<doca_gpu_dev_verbs_qp *>(qp);
}

__device__ __forceinline__ doca_gpu_dev_verbs_addr address(uint64_t addr,
								    uint32_t key)
{
	return doca_gpu_dev_verbs_addr{addr, key};
}

template <bool IsRead, bool WithMcst>
__device__ __forceinline__ uint64_t post_rdma(
	doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_addr remote,
	doca_gpu_dev_verbs_addr local, uint64_t length,
	doca_gpu_dev_verbs_addr dump)
{
	const uint64_t chunks = (length + DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE - 1) /
		DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
	const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<kSharing>(
		qp, chunks + (WithMcst ? 1 : 0));
	uint64_t remaining = length;
	uint64_t last = base;

	for (uint64_t i = 0; i < chunks; ++i) {
		last = base + i;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		const uint32_t bytes = static_cast<uint32_t>(
			remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				: remaining);
		if (IsRead) {
			doca_gpu_dev_verbs_wqe_prepare_read(
				qp, wqe, last, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
				remote.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				remote.key,
				local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				local.key, bytes);
		} else {
			doca_gpu_dev_verbs_wqe_prepare_write(
				qp, wqe, last, DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
				DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, 0,
				remote.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				remote.key,
				local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
				local.key, bytes);
		}
		remaining -= bytes;
	}
	if (WithMcst) {
		last = base + chunks;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		doca_gpu_dev_verbs_wqe_prepare_dump(
			qp, wqe, last, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
			dump.addr, dump.key, 1);
	}
	doca_gpu_dev_verbs_mark_wqes_ready<kSharing>(qp, base, last);
	doca_gpu_dev_verbs_submit<kSharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(qp, last + 1);
	return last;
}

__device__ __forceinline__ uint64_t post_send(
	doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_addr local,
	uint64_t length)
{
	const uint64_t chunks = (length + DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE - 1) /
		DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE;
	const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<kSharing>(qp, chunks);
	uint64_t remaining = length;
	uint64_t last = base;
	for (uint64_t i = 0; i < chunks; ++i) {
		last = base + i;
		auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, last);
		const uint32_t bytes = static_cast<uint32_t>(
			remaining > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
				: remaining);
		doca_gpu_dev_verbs_wqe_prepare_send(
			qp, wqe, last, DOCA_GPUNETIO_MLX5_OPCODE_SEND,
			DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, 0,
			local.addr + i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE,
			local.key, bytes);
		remaining -= bytes;
	}
	doca_gpu_dev_verbs_mark_wqes_ready<kSharing>(qp, base, last);
	doca_gpu_dev_verbs_submit<kSharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(qp, last + 1);
	return last;
}

} // namespace

extern "C" __device__ uint64_t
rdma4py_gpunetio_put(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			 uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_rdma<false, false>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_get(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			 uint64_t local_addr, uint32_t lkey, uint64_t length)
{
	return post_rdma<true, false>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(0, 0));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_get_mcst(uint64_t qp, uint64_t remote_addr, uint32_t rkey,
			      uint64_t local_addr, uint32_t lkey, uint64_t length,
			      uint64_t dump_addr, uint32_t dump_lkey)
{
	return post_rdma<true, true>(
		qp_from_u64(qp), address(remote_addr, rkey),
		address(local_addr, lkey), length, address(dump_addr, dump_lkey));
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_send(uint64_t qp, uint64_t local_addr, uint32_t lkey,
			  uint64_t length)
{
	return post_send(qp_from_u64(qp), address(local_addr, lkey), length);
}

extern "C" __device__ uint64_t
rdma4py_gpunetio_recv(uint64_t qp, uint64_t local_addr, uint32_t lkey,
			  uint64_t length)
{
	auto *device_qp = qp_from_u64(qp);
	const uint64_t ticket = doca_gpu_dev_verbs_reserve_wq_slots<
		kSharing, DOCA_GPUNETIO_VERBS_QP_RQ>(device_qp, 1);
	auto *wqe = doca_gpu_dev_verbs_get_rwqe_ptr(device_qp, ticket);
	doca_gpu_dev_verbs_wqe_prepare_recv(
		device_qp, wqe, local_addr, lkey, static_cast<uint32_t>(length));
	doca_gpu_dev_verbs_mark_wqes_ready<
		kSharing, DOCA_GPUNETIO_VERBS_QP_RQ>(device_qp, ticket, ticket);
	doca_gpu_dev_verbs_submit<kSharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler,
		DOCA_GPUNETIO_VERBS_QP_RQ>(device_qp, ticket + 1);
	return ticket;
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_send(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_test_send(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_one_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_test_recv(uint64_t qp, uint64_t ticket)
{
	return doca_gpu_dev_verbs_poll_one_cq_at<kSharing,
		DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(qp_from_u64(qp)), ticket);
}

extern "C" __device__ int32_t
rdma4py_gpunetio_wait_recv_mcst(uint64_t qp, uint64_t ticket,
				    uint64_t dump_addr, uint32_t dump_lkey)
{
	auto *device_qp = qp_from_u64(qp);
	const uint64_t dump_ticket = doca_gpu_dev_verbs_reserve_wq_slots<kSharing>(
		device_qp, 1);
	auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(device_qp, dump_ticket);
	doca_gpu_dev_verbs_wqe_prepare_dump(
		device_qp, wqe, dump_ticket,
		DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, dump_addr, dump_lkey, 1);
	doca_gpu_dev_verbs_mark_wqes_ready<kSharing>(
		device_qp, dump_ticket, dump_ticket);
	doca_gpu_dev_verbs_submit<kSharing,
		DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, kHandler>(
		device_qp, dump_ticket + 1);
	int status = doca_gpu_dev_verbs_poll_cq_at<
		kSharing, DOCA_GPUNETIO_VERBS_QP_SQ>(
		doca_gpu_dev_verbs_qp_get_cq_sq(device_qp), dump_ticket);
	if (status != 0)
		return status;
	return doca_gpu_dev_verbs_poll_cq_at<
		kSharing, DOCA_GPUNETIO_VERBS_QP_RQ>(
		doca_gpu_dev_verbs_qp_get_cq_rq(device_qp), ticket);
}
