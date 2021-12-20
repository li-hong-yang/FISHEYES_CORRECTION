/*********************************
头文件
**********************************/

#include "utils.h"


int get_filenames(const std::string& dir, std::vector<std::string>& filenames)
{
    fs::path path(dir);
    if (!fs::exists(path))
    {
        return -1;
    }
 
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter!=end_iter; ++iter)
    {
        if (fs::is_regular_file(iter->status()))
        {
            filenames.push_back(iter->path().string());
        }
 
        if (fs::is_directory(iter->status()))
        {
            get_filenames(iter->path().string(), filenames);
        }
    }
 
    return filenames.size();
}

