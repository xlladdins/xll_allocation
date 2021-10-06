// MSK.h - Mosek wrappers
#pragma once
#include <stdexcept>
#include <mosek.h>

namespace MSK {

	class env {
		MSKenv_t e;
	public:
		env()
			: e(NULL)
		{
			MSKrescodee res = MSK_makeenv(&e, NULL);
			if (MSK_RES_OK != res) {
				char symname[MSK_MAX_STR_LEN];
				char desc[MSK_MAX_STR_LEN];
				MSK_getcodedesc(res, symname, desc);

				throw std::runtime_error(desc);
			}
		}
		env(const env&) = delete;
		env& operator=(const env&) = delete;
		~env()
		{
			if (e) {
				MSK_deleteenv(&e);
			}
		}
		operator MSKenv_t()
		{
			return e;
		}
	};
	class task {
		env e;
		MSKtask_t t;
	public:
		task(MSKint32t maxnumcon, MSKint32t maxnumvar)
			: t(NULL)
		{
			MSKrescodee res = MSK_maketask(e, maxnumcon, maxnumvar, &t);
			if (MSK_RES_OK != res) {
				char symname[MSK_MAX_STR_LEN];
				char desc[MSK_MAX_STR_LEN];
				MSK_getcodedesc(res, symname, desc);

				throw std::runtime_error(desc);
			}
		}
		task(const task&) = delete;
		task& operator=(const task&) = delete;
		~task()
		{
			if (e) {
				MSK_deletetask(&t);
			}
		}
		operator MSKenv_t()
		{
			return e;
		}
	};

} // namespace MSK
