#ifndef RDMA4PY_MLX5DV_BRIDGE_H
#define RDMA4PY_MLX5DV_BRIDGE_H

#include <stdint.h>
#include <string.h>

#include <infiniband/mlx5dv.h>

struct rdma4py_mlx5_qp_info {
	void *sq_buf;
	uint32_t sq_wqe_cnt;
	uint32_t sq_stride;
	void *rq_buf;
	uint32_t rq_wqe_cnt;
	uint32_t rq_stride;
	uint32_t *sq_dbrec;
	uint32_t *rq_dbrec;
	uint64_t *uar;
	uint32_t uar_size;
};

struct rdma4py_mlx5_cq_info {
	void *buf;
	uint32_t *dbrec;
	uint32_t cqe_cnt;
	uint32_t cqe_size;
	uint32_t cqn;
};

typedef int (*rdma4py_mlx5dv_init_obj_fn)(struct mlx5dv_obj *, uint64_t);

static inline int
rdma4py_mlx5dv_qp_info(void *init_obj_fn, struct ibv_qp *qp,
			struct rdma4py_mlx5_qp_info *info)
{
	struct mlx5dv_qp dv_qp;
	struct mlx5dv_obj obj;
	int rc;

	memset(&dv_qp, 0, sizeof(dv_qp));
	memset(&obj, 0, sizeof(obj));
	obj.qp.in = qp;
	obj.qp.out = &dv_qp;
	rc = ((rdma4py_mlx5dv_init_obj_fn)init_obj_fn)(&obj, MLX5DV_OBJ_QP);
	if (rc != 0)
		return rc;

	info->sq_buf = dv_qp.sq.buf;
	info->sq_wqe_cnt = dv_qp.sq.wqe_cnt;
	info->sq_stride = dv_qp.sq.stride;
	info->rq_buf = dv_qp.rq.buf;
	info->rq_wqe_cnt = dv_qp.rq.wqe_cnt;
	info->rq_stride = dv_qp.rq.stride;
	info->sq_dbrec = (uint32_t *)dv_qp.dbrec + MLX5_SND_DBR;
	info->rq_dbrec = (uint32_t *)dv_qp.dbrec + MLX5_RCV_DBR;
	info->uar = (uint64_t *)dv_qp.bf.reg;
	info->uar_size = dv_qp.bf.size;
	return 0;
}

static inline int
rdma4py_mlx5dv_cq_info(void *init_obj_fn, struct ibv_cq *cq,
			struct rdma4py_mlx5_cq_info *info)
{
	struct mlx5dv_cq dv_cq;
	struct mlx5dv_obj obj;
	int rc;

	memset(&dv_cq, 0, sizeof(dv_cq));
	memset(&obj, 0, sizeof(obj));
	obj.cq.in = cq;
	obj.cq.out = &dv_cq;
	rc = ((rdma4py_mlx5dv_init_obj_fn)init_obj_fn)(&obj, MLX5DV_OBJ_CQ);
	if (rc != 0)
		return rc;

	info->buf = dv_cq.buf;
	info->dbrec = (uint32_t *)dv_cq.dbrec;
	info->cqe_cnt = dv_cq.cqe_cnt;
	info->cqe_size = dv_cq.cqe_size;
	info->cqn = dv_cq.cqn;
	return 0;
}

#endif
